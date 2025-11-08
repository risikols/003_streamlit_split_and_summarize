import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

st.set_page_config(page_title="Resumen de PDFs", layout="wide")
st.title("Resumen de documentos con LangChain y Streamlit")

# Configura tu clave desde Secrets de Streamlit
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Por favor configura la variable de entorno OPENAI_API_KEY en Secrets")
    st.stop()
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Subida de PDF
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")
if uploaded_file:
    try:
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

        if not text.strip():
            st.warning("No se pudo extraer texto del PDF.")
            st.stop()

        # Dividir texto en fragmentos
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_text(text)
        st.write(f"Texto dividido en {len(chunks)} fragmentos")

        # Crear embeddings y vectorstore
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # Cadena de QA
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # Consulta del usuario
        query = st.text_input("Escribe tu pregunta sobre el PDF:")
        if query:
            try:
                answer = qa.run(query)
                st.markdown(f"**Respuesta:** {answer}")
            except Exception as e:
                st.error(f"Ocurrió un error al procesar tu pregunta: {str(e)}")

    except Exception as e:
        st.error(f"❌ Ocurrió un error procesando el PDF: {str(e)}")
