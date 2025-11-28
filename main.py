import streamlit as st
from PyPDF2 import PdfReader
import openai
from openai.error import OpenAIError, RateLimitError, InvalidRequestError

st.set_page_config(page_title="PDF Summarizer", layout="wide")
st.title("📝 PDF Summarizer con GPT (Tokens opcionales)")

st.markdown("""
- Introduce tu API Key de OpenAI para generar resúmenes reales.
- Si no hay API Key o no hay tokens disponibles, la app generará un resumen simulado.
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
                if openai_api_key and openai_api_key.strip():
                    try:
                        openai.api_key = openai_api_key
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
                        st.subheader("Resumen generado")
                        st.write(summary)

                    except RateLimitError:
                        st.warning("No hay suficientes tokens en tu cuenta de OpenAI.")
                    except InvalidRequestError as e:
                        # Esto captura errores por clave inválida o texto muy largo
                        st.warning(f"Error con la solicitud a OpenAI: {e}")
                    except OpenAIError as e:
                        st.warning(f"Error de OpenAI: {e}")
                    except Exception as e:
                        st.error(f"Error inesperado: {e}")
                else:
                    # Resumen simulado
                    lines = text.strip().split("\n")
                    summary = "\n".join(lines[:3])
                    st.subheader("Resumen simulado")
                    st.write(summary)
                    st.info("Introduce tu API Key válida para generar resúmenes reales y consumir tokens.")
