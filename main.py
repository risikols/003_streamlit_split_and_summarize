# main_streamlit_progress.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from openai.error import OpenAIError
import time

# Inicializar cliente OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="PDF Split & Summarize", layout="wide")
st.title("📄 PDF Split & Summarize (con progreso)")

uploaded_file = st.file_uploader("Sube tu PDF", type=["pdf"])

def summarize_text(text_chunk, model="gpt-3.5-turbo"):
    """
    Genera un resumen de un fragmento de texto usando OpenAI
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un asistente que resume texto largo en resúmenes claros y concisos."},
                {"role": "user", "content": text_chunk}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        st.error(f"Error con OpenAI: {e}")
        return None

if uploaded_file:
    # Leer PDF
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"

    # Dividir en fragmentos
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n", " "]
    )
    chunks = splitter.split_text(text)
    st.write(f"Texto dividido en {len(chunks)} fragmentos.")

    # Crear barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Map step: resumir cada fragmento
    summaries = []
    for i, chunk in enumerate(chunks):
        status_text.text(f"Procesando fragmento {i+1}/{len(chunks)}...")
        summary = summarize_text(chunk, model="gpt-3.5-turbo")
        if summary:
            summaries.append(summary)
        progress_bar.progress((i + 1) / len(chunks))
        time.sleep(0.1)  # pequeño delay para que la barra se actualice

    # Reduce step: resumir los resúmenes
    if summaries:
        combined_text = "\n".join(summaries)
        st.write("Generando resumen final...")
        final_summary = summarize_text(combined_text, model="gpt-3.5-turbo")
        st.success("✅ Resumen final generado")
        st.text_area("Resumen Final", final_summary, height=300)

    progress_bar.empty()
    status_text.empty()
