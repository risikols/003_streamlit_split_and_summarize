import streamlit as st
import os
from PyPDF2 import PdfReader

# Manejo de errores en caso de dependencias faltantes
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI
except ImportError as e:
    st.error(f"Falta una dependencia: {e}. Revisa tu requirements.txt")
    st.stop()

# Configuración API
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Por favor configura la variable de entorno OPENAI_API_KEY en Streamlit Secrets")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.title("Resumen de documentos con LangChain y Streamlit")

uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file:
    try:
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        if not text:
            st.warning("No se pudo extraer texto del PDF")
            st.stop()

        # Dividir texto en fragmentos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)

        # Crear embeddings y base vectorial
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # Crear cadena de QA
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # Consulta al usuario
        query = st.text_input("Escribe tu pregunta sobre el PDF:")
        if query:
            answer = qa.run(query)
            st.write(answer)

    except Exception as e:
        st.error(f"❌ Ocurrió un error procesando el PDF: {e}")

