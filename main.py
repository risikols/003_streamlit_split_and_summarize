import streamlit as st
from PyPDF2 import PdfReader

# Configuración de la página
st.set_page_config(page_title="PDF Summarizer (Simulado)", layout="wide")
st.title("📝 PDF Summarizer con Simulación (sin tokens)")

# Subida de archivo PDF
uploaded_file = st.file_uploader("Sube tu PDF aquí", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if not text.strip():
        st.error("No se encontró texto en el PDF.")
    else:
        st.subheader("Texto extraído")
        st.text_area("Contenido del PDF", text, height=300)

        if st.button("Generar resumen simulado"):
            with st.spinner("Generando resumen simulado..."):
                # Aquí no llamamos a OpenAI, hacemos un resumen de prueba
                # Por ejemplo, tomamos las primeras 3 líneas
                lines = text.strip().split("\n")
                summary = "\n".join(lines[:3])
                
                st.subheader("Resumen simulado")
                st.write(summary)
                st.info("Este resumen es simulado. No se han consumido tokens de OpenAI.")
