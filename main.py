import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import os

st.set_page_config(page_title="PDF Split & Summarize", layout="wide")

# Configuración de API Key
if "OPENAI_API_KEY" not in st.secrets:
    st.warning("Please set your OpenAI API key in Streamlit secrets.")
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")

st.title("PDF Split & Summarize")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"

    st.subheader("PDF content loaded")
    st.write(text[:500] + "...")  # mostrar solo los primeros 500 caracteres

    # Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    st.write(f"Total chunks created: {len(chunks)}")

    # Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # QA Chain
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    query = st.text_input("Ask a question about your PDF:")

    if query:
        docs = vectorstore.similarity_search(query)
        answer = qa_chain.run(input_documents=docs, question=query)
        st.subheader("Answer")
        st.write(answer)
