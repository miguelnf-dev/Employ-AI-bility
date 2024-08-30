import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from utils import initialize_vector_store, setup_retrieval_chain

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