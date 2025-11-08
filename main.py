import os
import streamlit as st
from PyPDF2 import PdfReader

# Importaciones LangChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Configuración de la API de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Por favor configura la variable de entorno OPENAI_API_KEY")
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.title("Resumen de documentos con LangChain y Streamlit")

# Carga de PDF
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")
if uploaded_file:
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""

    # Dividir el texto en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)

    # Crear embeddings y base vectorial
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Crear cadena de QA
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY),
        c
