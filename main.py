import streamlit as st
from PyPDF2 import PdfReader
import openai  # No se importa OpenAIError

st.set_page_config(page_title="PDF Summarizer", layout="wide")
st.title("📝 PDF Summarizer con GPT (real o simulado)")

st.markdown("""
- Introduce tu API Key de OpenAI para generar resúmenes reales.
- Si no hay API Key o no se puede usar, la app generará un **resumen simulado**.
""")

# Input opcional de API Key
openai_api_key = st.text_input(
    "Tu OpenAI API Key (opcional)",
    type="password",
    placeholder="sk-XXXX..."
)

# Subida de PDF
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

        if st.button("Generar resumen"):
            with st.spinner("Generando resumen..."):

                # Si la clave está presente, intenta usar la API
                if openai_api_key and openai_api_key.strip():
                    try:
                        openai.api_key = openai_api_key
                        response = openai.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "Eres un asistente útil que resume textos."},
                                {"role": "user", "content": f"Resume este texto:\n{text}"}
                            ],
                            max_tokens=500,
                            temperature=0.5,
                        )
                        summary = response.choices[0].message.content
                        st.subheader("Resumen generado")
                        st.write(summary)

                    except Exception as e:
                        # Control genérico de errores de OpenAI
                        msg = str(e)
                        if "insufficient_quota" in msg or "Rate limit" in msg:
                            st.warning("No hay suficientes tokens en tu cuenta de OpenAI.")
                        else:
                            st.warning(f"Error de OpenAI: {msg}")

                        # Resumen simulado
                        summary = "\n".join(text.strip().split("\n")[:3])
                        st.subheader("Resumen simulado")
                        st.write(summary)

                else:
                    # Resumen simulado si no hay API Key
                    summary = "\n".join(text.strip().split("\n")[:3])
                    st.subheader("Resumen simulado")
                    st.write(summary)
                    st.info("Introduce tu API Key válida para generar resúmenes reales y consumir tokens.")
generar resúmenes reales y consumir tokens.")
