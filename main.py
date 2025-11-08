import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from io import StringIO

# ------------------------------
# LLM and API key loading function
# ------------------------------
def load_LLM(openai_api_key):
    """
    Crea un modelo de lenguaje usando ChatOpenAI.
    Compatible con Python 3.11 y LangChain 1.0+
    """
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo"  # Cambiar a "gpt-4" si tu API key lo permite
    )
    return llm

# ------------------------------
# Página y encabezado
# ------------------------------
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")

# Intro: instrucciones
col1, col2 = st.columns(2)
with col1:
    st.markdown("ChatGPT no puede resumir textos muy largos. Ahora puedes hacerlo con esta app.")

with col2:
    st.write("Contacta con [AI Accelera](https://aiaccelera.com) para tus proyectos de IA")

# ------------------------------
# Input de OpenAI API Key
# ------------------------------
st.markdown("## Ingresa tu OpenAI API Key")

def get_openai_api_key():
    input_text = st.text_input(
        label="OpenAI API Key",
        placeholder="Ex: sk-...",
        key="openai_api_key_input",
        type="password"
    )
    return input_text

openai_api_key = get_openai_api_key()

# ------------------------------
# Input del archivo de texto
# ------------------------------
st.markdown("## Sube el archivo de texto que quieres resumir")
uploaded_file = st.file_uploader("Elige un archivo", type="txt")

# ------------------------------
# Output: Resumen
# ------------------------------
st.markdown("### Aquí está tu resumen:")

if uploaded_file is not None:
    # Leer archivo como string
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    file_input = stringio.read()

    # Validación de longitud
    if len(file_input.split(" ")) > 20000:
        st.write("Por favor ingresa un archivo más corto. Máximo 20,000 palabras.")
        st.stop()

    # Validar API Key
    if not openai_api_key:
        st.warning(
            'Por favor ingresa tu OpenAI API Key. Instrucciones [aquí](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)',
            icon="⚠️"
        )
        st.stop()

    # Dividir texto en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=5000,
        chunk_overlap=350
    )
    splitted_documents = text_splitter.create_documents([file_input])

    # Cargar LLM
    llm = load_LLM(openai_api_key=openai_api_key)

    # Cargar chain de resumen
    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce"
    )

    # Ejecutar chain
    summary_output = summarize_chain.run(splitted_documents)

    st.write(summary_output)
