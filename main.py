import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import openai
import os

# Configuración de la API de OpenAI desde Secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Por favor configura la variable de entorno OPENAI_API_KEY en Secrets")
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.title("Resumen de documentos PDF con LangChain y Streamlit")

uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file:
    try:
        # Leer PDF
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        st.success(f"Texto extraído del PDF ({len(text.split())} palabras)")

        # Dividir texto en fragmentos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)
        st.write(f"Texto dividido en {len(chunks)} fragmentos")

        # Crear embeddings y vectorstore
        try:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embeddings)
        except openai.OpenAIError as e:
            st.error(f"Error de OpenAI al generar embeddings: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error inesperado al crear vectorstore: {e}")
            st.stop()

        # Crear cadena de QA
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # Input de usuario
        query = st.text_input("Escribe tu pregunta sobre el PDF:")
        if query:
            try:
                answer = qa.run(query)
                st.write("**Respuesta:**", answer)
            except openai.OpenAIError as e:
                st.error(f"Error de OpenAI al procesar la consulta: {e}")
            except Exception as e:
                st.error(f"Error inesperado al procesar la consulta: {e}")

    except Exception as e:
        st.error(f"❌ Ocurrió un error procesando el PDF: {e}")
