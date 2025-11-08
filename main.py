import streamlit as st
from openai import OpenAI, OpenAIError
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuración del cliente OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # usa tu API key en Streamlit secrets

st.set_page_config(page_title="PDF Split & Summarize", layout="wide")
st.title("📄 PDF Split & Summarize con OpenAI y LangChain")

# Subida de PDF
uploaded_file = st.file_uploader("Sube tu PDF", type=["pdf"])

if uploaded_file is not None:
    try:
        # Leer PDF
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        st.success(f"PDF cargado con {len(reader.pages)} páginas.")

        # Dividir el texto en fragmentos
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,   # tamaño de cada fragmento
            chunk_overlap=100  # solapamiento entre fragmentos
        )
        texts = splitter.split_text(text)
        st.write(f"Texto dividido en {len(texts)} fragmentos.")

        # Botón para resumir cada fragmento
        if st.button("📝 Resumir PDF"):
            summaries = []
            progress = st.progress(0)
            for i, chunk in enumerate(texts):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "Eres un asistente que resume texto."},
                            {"role": "user", "content": f"Resume este texto:\n\n{chunk}"}
                        ],
                        temperature=0.5
                    )
                    summary = response.choices[0].message.content
                    summaries.append(summary)
                except OpenAIError as e:
                    st.error(f"Error en fragmento {i+1}: {e}")
                    summaries.append("[Error al generar resumen]")
                progress.progress((i + 1) / len(texts))

            # Mostrar resumen final
            final_summary = "\n\n".join(summaries)
            st.subheader("Resumen completo del PDF")
            st.write(final_summary)

    except Exception as e:
        st.error(f"❌ Error procesando el PDF: {e}")
