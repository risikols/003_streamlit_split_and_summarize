import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Configuración de la app
st.set_page_config(page_title="Split and Summarize", layout="wide")

st.title("📄 Split & Summarize Documents")
st.write("Sube un documento, divídelo en partes y genera un resumen con LLM.")

# Cargar archivo
uploaded_file = st.file_uploader("Selecciona un archivo de texto (.txt)", type="txt")

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    
    # Selector de tamaño de chunk
    chunk_size = st.slider("Tamaño del chunk (caracteres)", 1000, 5000, 2000)
    
    # Dividir el texto en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    
    st.write(f"El documento se dividió en **{len(chunks)}** partes.")
    
    # Mostrar primeros 2 chunks como ejemplo
    for i, chunk in enumerate(chunks[:2]):
        st.subheader(f"Chunk {i+1}")
        st.write(chunk)
    
    # Resumen con LLM
    st.subheader("Resumen del Documento")
    
    # Ingresar API Key de OpenAI
    openai_api_key = st.text_input("Introduce tu OpenAI API Key", type="password")
    
    if openai_api_key:
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
        
        # Plantilla de resumen
        template = """Resume el siguiente texto en español de manera clara y concisa:

        {text_chunk}
        """
        prompt = PromptTemplate(input_variables=["text_chunk"], template=template)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Generar resúmenes por chunk y combinarlos
        summaries = []
        for chunk in chunks:
            summary = chain.run(text_chunk=chunk)
            summaries.append(summary)
        
        final_summary = "\n\n".join(summaries)
        
        st.text_area("Resumen final", value=final_summary, height=300)
