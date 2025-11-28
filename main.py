import streamlit as st
from PyPDF2 import PdfReader
import openai

# Configurar API Key desde Secrets de Streamlit
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Configuración de la página
st.set_page_config(page_title="PDF Summarizer", layout="wide")
st.title("📝 PDF Summarizer con GPT-3.5/4")

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

        if st.button("Generar resumen"):
            with st.spinner("Generando resumen..."):
                try:
                    response = client.chat.completions.create(
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
                    st.error(f"Ocurrió un error al generar el resumen: {e}")

                except Exception as e:
                    st.error(f"Ocurrió un error al generar el resumen: {e}")

