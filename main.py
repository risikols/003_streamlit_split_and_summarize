import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import os

# --- Configuración API Key ---
if "OPENAI_API_KEY" not in st.secrets:
    st.warning("Por favor agrega tu OPENAI_API_KEY en Secrets de Streamlit Cloud")
else:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("Resumen de documentos con LangChain y Streamlit")

# --- Subida de PDF ---
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""

    if not text:
        st.error("No se pudo extraer texto del PDF")
    else:
        # --- Spinner mientras procesa ---
        with st.spinner("Procesando PDF y generando embeddings..."):
            # Dividir el texto en fragmentos pequeños para no colgar la app
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # chunks más pequeños
                chunk_overlap=50
            )
            chunks = text_splitter.split_text(text[:10000])  # prueba inicial con primeros 10k caracteres

            # Crear embeddings y base vectorial
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embeddings)

            # Crear cadena de QA
            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )

        st.success("PDF procesado ✅")

        # --- Consulta del usuario ---
        query = st.text_input("Escribe tu pregunta sobre el PDF:")
        if query:
            with st.spinner("Buscando respuesta..."):
                answer = qa.run(query)
            st.write(answer)
