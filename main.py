import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader
import os

# Configuración del API Key de OpenAI
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Por favor configura tu API Key de OpenAI en los secretos de Streamlit")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("Split and Summarize PDF")

# Subir archivo PDF
uploaded_file = st.file_uploader("Sube tu PDF", type=["pdf"])
if uploaded_file:
    # Guardar temporalmente el PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Cargar PDF
    loader = UnstructuredPDFLoader("temp.pdf")
    documents = loader.load()

    st.write(f"Documentos cargados: {len(documents)} páginas")

    # Dividir el texto en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    st.write(f"Documentos divididos en chunks: {len(docs)}")

    # Crear embeddings y vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Crear chain de QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # Interfaz de consulta
    query = st.text_input("Pregunta sobre el documento:")
    if query:
        answer = qa_chain.run(query)
        st.write("Respuesta:")
        st.write(answer)
