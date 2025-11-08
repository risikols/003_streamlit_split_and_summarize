import streamlit as st
import regex as re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Configuración de la app
st.set_page_config(page_title="Split & Summarize", layout="wide")
st.title("Split & Summarize Documents")

# Sidebar
st.sidebar.header("Configuración")
openai_api_key = st.sidebar.text_input("API Key de OpenAI", type="password")

# Subida de documentos
uploaded_files = st.file_uploader("Sube tus documentos (PDF, TXT, DOCX)", accept_multiple_files=True)

if uploaded_files and openai_api_key:
    texts = []
    for file in uploaded_files:
        # Lectura simple de TXT
        try:
            content = file.read().decode("utf-8", errors="ignore")
            texts.append(content)
        except Exception as e:
            st.error(f"No se pudo leer {file.name}: {e}")

    # Dividir el texto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.create_documents(texts)

    # Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Crear chain de QA
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    # Pregunta
    query = st.text_input("Escribe tu pregunta sobre los documentos")
    if query:
        answer = qa.run(query)
        st.write("**Respuesta:**", answer)
else:
    st.info("Sube archivos y proporciona tu API Key de OpenAI para empezar.")
