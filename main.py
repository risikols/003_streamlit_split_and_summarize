import streamlit as st
import os
from PyPDF2 import PdfReader

# LangChain imports compatibles con 0.0.285
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Configuración de la API Key de OpenAI
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Por favor configura la variable de entorno OPENAI_API_KEY en Streamlit Secrets")
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.title("Resumen de documentos PDF con LangChain y Streamlit")

# Subida de PDF
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""

    if not text.strip():
        st.warning("No se pudo extraer texto del PDF.")
    else:
        # Dividir texto en fragmentos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)

        # Crear embeddings y vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # Crear cadena de QA
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # Input de consulta
        query = st.text_input("Escribe tu pregunta sobre el PDF:")
        if query:
            with st.spinner("Generando respuesta..."):
                answer = qa.run(query)
                st.write(answer)
