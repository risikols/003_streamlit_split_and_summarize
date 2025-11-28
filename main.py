import streamlit as st
from PyPDF2 import PdfReader
import openai

st.set_page_config(page_title="PDF Summarizer", layout="wide")
st.title("📝 PDF Summarizer con GPT-3.5")

# Cargar API Key desde Streamlit Secrets
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# Subida de archivo
uploaded_file = st.file_uploader("Sube tu PDF o TXT aquí", type=["pdf", "txt"])

def extract_text(file):
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
                    # Intentar llamada real a OpenAI
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Eres un asistente útil que resume textos."},
                            {"role": "user", "content": f"Resume este texto:\n{text}"}
                        ],
                        max_tokens=500,
                        temperature=0.5,
                    )
                    summary = response['choices'][0]['message']['content']
                    st.subheader("Resumen generado (real)")
                    st.write(summary)
                except openai.error.InvalidRequestError as e:
                    # Si la API devuelve insufficient_quota
                    if "insufficient_quota" in str(e):
                        st.warning("No hay suficientes tokens disponibles en tu cuenta de OpenAI. Se mostrará un resumen simulado.")
                        summary = text[:500] + "..." if len(text) > 500 else text
                        st.subheader("Resumen simulado")
                        st.write(summary)
                    else:
                        st.error(f"Ocurrió un error en la API: {e}")
                except Exception as e:
                    # Otros errores de OpenAI
                    st.warning("No se pudo generar un resumen real. Se mostrará un resumen simulado.")
                    summary = text[:500] + "..." if len(text) > 500 else text
                    st.subheader("Resumen simulado")
                    st.write(summary)
