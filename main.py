import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
import os
import tempfile
import regex

# Configuración de la API de OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Por favor, configura la variable de entorno OPENAI_API_KEY")
    st.stop()

# Título
st.title("Split & Summarize")

# Cargar archivo
uploaded_file = st.file_uploader("Sube un archivo TXT o PDF", type=["txt", "pdf"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Leer texto del archivo
    if uploaded_file.type == "text/plain":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

    # Dividir el texto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_text(text)

    # Crear embeddings y base de datos FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(docs, embeddings)

    # Configurar LLM
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    # Cadena de resumen
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    # Ejecutar resumen
    summary = chain.run(docs)
    st.subheader("Resumen")
    st.write(summary)
)
