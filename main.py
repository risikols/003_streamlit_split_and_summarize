import streamlit as st
from PyPDF2 import PdfReader
import openai

# Configura tu API key desde secrets
openai.api_key = st.secrets.get("OPENAI_API_KEY")

st.set_page_config(page_title="Split and Summarize", layout="wide")
st.title("📄 PDF Split & Summarize")

# Subir archivo PDF
uploaded_file = st.file_uploader("Sube un PDF", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    st.subheader("Contenido del PDF")
    st.text_area("Texto extraído", text, height=300)

    if st.button("Generar resumen"):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente que resume documentos."},
                    {"role": "user", "content": text}
                ],
                max_tokens=500
            )
            resumen = response.choices[0].message.content
            st.subheader("Resumen")
            st.write(resumen)
        except Exception as e:
            st.error(f"Ocurrió un error al generar el resumen: {e}")

