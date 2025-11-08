import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Configurar API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Por favor configura la variable de entorno OPENAI_API_KEY")
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.title("Resumen de documentos con LangChain y Streamlit")

# Subida de PDF
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""

    if not text.strip():
        st.warning("El PDF no contiene texto legible.")
    else:
        # Dividir texto
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)

        # Crear embeddings y vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # Configurar QA
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # Consulta usuario
        query = st.text_input("Escribe tu pregunta sobre el PDF:")
        if query:
            answer = qa.run(query)
            st.write(answer)
