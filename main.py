# main.py
import streamlit as st
import PyPDF2
import openai
from openai.error import OpenAIError
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configurar API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("📄 PDF Split & Summarize")

uploaded_file = st.file_uploader("Sube tu archivo PDF", type=["pdf"])

def extract_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text(text, chunk_size=1000, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def summarize_chunk(chunk):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Resume este texto:\n{chunk}"}],
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except OpenAIError as e:
        st.error(f"Error procesando el fragmento: {e}")
        return None

if uploaded_file:
    text = extract_text(uploaded_file)
    if not text.strip():
        st.error("No se pudo extraer texto del PDF.")
    else:
        st.info("Dividiendo el texto en fragmentos...")
        chunks = split_text(text)
        st.success(f"Texto dividido en {len(chunks)} fragmentos.")

        st.info("Generando resúmenes de cada fragmento...")
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            st.write(f"Procesando fragmento {i}/{len(chunks)}...")
            summary = summarize_chunk(chunk)
            if summary:
                summaries.append(summary)

        if summaries:
            st.info("Generando resumen final...")
            final_summary = summarize_chunk(" ".join(summaries))
            st.subheader("Resumen final")
            st.write(final_summary)
