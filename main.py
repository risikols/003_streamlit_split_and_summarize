import streamlit as st
import os
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# ---------------------------
# Configuración API
# ---------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ Por favor configura tu clave OPENAI_API_KEY en Streamlit Secrets.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.title("📄 Resumen de PDF con LangChain y Streamlit")

# ---------------------------
# Subida de archivo
# ---------------------------
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file:
    try:
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            st.warning("⚠️ No se pudo extraer texto del PDF.")
            st.stop()

        # ---------------------------
        # Dividir texto en chunks
        # ---------------------------
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)

        # ---------------------------
        # Crear embeddings y vectorstore
        # ---------------------------
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # ---------------------------
        # Configurar QA
        # ---------------------------
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # ---------------------------
        # Consulta del usuario
        # ---------------------------
        query = st.text_input("Escribe tu pregunta sobre el PDF:")
        if query:
            with st.spinner("⌛ Generando respuesta..."):
                answer = qa.run(query)
                st.markdown(f"**Respuesta:** {answer}")

    except Exception as e:
        st.error(f"❌ Ocurrió un error procesando el PDF: {e}")

