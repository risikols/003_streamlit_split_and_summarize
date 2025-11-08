import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import openai
import os

# Configuración de la API de OpenAI desde Streamlit Secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Por favor configura la variable de entorno OPENAI_API_KEY en Streamlit Secrets.")
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.title("Resumen de documentos con LangChain y Streamlit")

uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file:
    try:
        # Leer el PDF
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

        if not text:
            st.warning("El PDF no contiene texto legible.")
        else:
            # Dividir el texto en fragmentos
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_text(text)
            st.success(f"Texto dividido en {len(chunks)} fragmentos")

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
                try:
                    answer = qa.run(query)
                    st.write(answer)
                except Exception as e:
                    st.error(f"❌ Error al procesar la consulta: {e}")

    except Exception as e:
        st.error(f"❌ Ocurrió un error procesando el PDF: {e}")
