import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# ----------------------------
# Configurar clave OpenAI
# ----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Por favor configura la variable de entorno OPENAI_API_KEY en Secrets")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.title("Resumen y QA de PDFs con LangChain y Streamlit")

# ----------------------------
# Subida de PDF
# ----------------------------
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file:
    try:
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        
        if not text:
            st.error("No se pudo extraer texto del PDF")
            st.stop()

        # ----------------------------
        # Dividir texto en fragmentos
        # ----------------------------
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Texto dividido en {len(chunks)} fragmentos")

        # ----------------------------
        # Crear embeddings y FAISS
        # ----------------------------
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # ----------------------------
        # Crear cadena QA
        # ----------------------------
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=OPENAI_API_KEY),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # ----------------------------
        # Interfaz de consulta
        # ----------------------------
        query = st.text_input("Escribe tu pregunta sobre el PDF:")
        if query:
            answer = qa.run(query)
            st.markdown(f"**Respuesta:** {answer}")

    except Exception as e:
        st.error(f"❌ Ocurrió un error procesando el PDF: {e}")
