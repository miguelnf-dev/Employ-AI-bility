import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from PIL import Image

# Load environment variables
load_dotenv()

# Set up file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "cv", "cv.pdf")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_huggingface")

mixtral = 'mixtral-8x7b-32768'
llama3_8b = 'llama3-8b-8192'
llama3_1 = 'llama-3.1-8b-instant'
gemma7b = 'gemma-7b-it'

# Initialize ChatGroq
llm = ChatGroq(model=llama3_8b)


# Load and process documents
@st.cache_resource
def load_and_process_documents():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please check the path.")
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter()
    docs = text_splitter.split_documents(documents)
    return docs

# Initialize embeddings and vector store
@st.cache_resource
def initialize_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    if os.path.exists(persistent_directory):
        # Load existing vector store
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        return db
    else:
        # No existing vector store, create a new one
        docs = load_and_process_documents()
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        return db


# Set up retriever and chains
@st.cache_resource
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
    
    # qa_system_prompt = (
    #     "You are an assistant for question-answering tasks about your Curriculum Vitae (i.e Miguel Fernandes)"
    #     "Use the following pieces of retrieved context to answer "
    #     "the question. If you don't know the answer, say that you "
    #     "don't know.If the questions are about anything else other than Miguel or the CV, say 'Thats a secret!'." 
    #     "Use three sentences maximum and keep the "
    #     "answer concise. Answer as if you were Miguel Fernandes, dont answer in the second or third person."
    #     "Dont hallucinate or make up information. I'll tip you 100$ if you get it right."
    #     "\n\n"
    #     "{context}"
    # )
    
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


def main():
    st.set_page_config(page_title="Employ-AI-bility", page_icon="🖥️")
    st.title("Chat with my CV 🤖")
    
    st.sidebar.title("About Miguel Fernandes")
    st.sidebar.markdown("""
    **Machine Learning Engineer** with a strong background in Data Science and AI. Experienced in developing innovative solutions in computer vision, NLP, and medical device technology.
    
    **Contact Information:**
    - Email: miguelnf3991@gmail.com
    - LinkedIn: [linkedin.com/in/miguelnevesfernandes](https://www.linkedin.com/in/miguelnevesfernandes/)
    - GitHub: [github.com/miguelnf-dev](https://github.com/miguelnf-dev)
    """)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    db = initialize_vector_store()
    
    rag_chain = setup_retrieval_chain(db)
    
    for message in st.session_state.chat_history:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.markdown(message.content)
    
    if prompt := st.chat_input("Ask a question about my CV:"):
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
            st.markdown(result["answer"])
        
        st.session_state.chat_history.append(SystemMessage(content=result["answer"]))

if __name__ == "__main__":
    main()


