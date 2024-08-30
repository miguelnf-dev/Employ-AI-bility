# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import hashlib
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Set up file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "cv", "cv.pdf")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_huggingface")
hash_file_path = os.path.join(db_dir, "cv_hash.txt")

llama3_8b = 'llama3-8b-8192'

# Initialize ChatGroq
llm = ChatGroq(model=llama3_8b)

def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()

def should_regenerate_database():
    if not os.path.exists(hash_file_path):
        return True
    
    with open(hash_file_path, "r") as f:
        stored_hash = f.read().strip()
    
    current_hash = get_file_hash(file_path)
    return stored_hash != current_hash

def update_hash_file():
    current_hash = get_file_hash(file_path)
    with open(hash_file_path, "w") as f:
        f.write(current_hash)

def load_and_process_documents():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please check the path.")
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter()
    docs = text_splitter.split_documents(documents)
    return docs

def initialize_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    if should_regenerate_database():
        print("Updating CV database... This may take a moment.")
        if os.path.exists(persistent_directory):
            import shutil
            shutil.rmtree(persistent_directory)
        
        docs = load_and_process_documents()
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        update_hash_file()
        print("CV database updated successfully!")
    else:
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    
    return db

def setup_retrieval_chain(_db):
    retriever = _db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    qa_system_prompt = ("""You are Miguel Fernandes's AI bot. You are an assistant for question-answering tasks. 
                        Use the following pieces of retrieved context to answer the question. 
                        If you don't know the answer, just say 'I dont have that information right now'. 
                        If its not related to Miguel or the CV, say 'Thats a secret!'.
                        Use three sentences maximum and keep the answer concise.
                        Speak in the first person as if you were Miguel Fernandes. 
                        For example. 'I am a Machine Learning Engineer with...'.
                        Analyze whats being asked and answer accordingly given the following context:
                    \n\n
                    {context}""")
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain