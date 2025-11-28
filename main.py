import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI, OpenAIError  # ✅ Import correcto para OpenAI >=1.0.0

st.set_page_config(page_title="PDF Summarizer", layout="wide")
st.title("📝 PDF/TXT Summarizer con GPT-3.5 / GPT-4")

# Inicializar cliente OpenAI
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

# Subida de archivo
uploaded_file = st.file_uploader("Sube tu PDF o TXT aquí", type=["pdf", "txt"])

def extract_text(file):
    """Extrae texto de PDF o TXT"""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    else:
        return file.getvalue().decode("utf-8").strip()

if uploaded_file:
    text = extract_text(uploaded_file)
    if not text:
        st.error("No se encontró texto en el archivo.")
    else:
        st.subheader("Texto extraído")
        st.text_area("Contenido del archivo", text, height=300)

        if st.button("Generar resumen"):
            with st.spinner("Generando resumen..."):
                summary = ""
                try:
                    # Llamada a la nueva API de OpenAI
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Eres un asistente útil que resume textos."},
                            {"role": "user", "content": f"Resume este texto:\n{text}"}
                        ],
                        max_tokens=500,
                        temperature=0.5
                    )
                    summary = response.choices[0].message.content
                    st.subheader("Resumen generado (real)")
                    st.write(summary)

                except OpenAIError as e:
                    # Control de error por falta de tokens
                    if "insufficient_quota" in str(e):
                        st.warning("No hay suficientes tokens disponibles en tu cuenta de OpenAI. Se mostrará un resumen simulado.")
                    else:
                        st.warning(f"No se pudo generar un resumen real: {e}")

                    # Resumen simulado
                    summary = text[:500] + "..." if len(text) > 500 else text
                    st.subheader("Resumen simulado")
                    st.write(summary)
