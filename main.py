import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# ===============================
# Configuración API Key OpenAI
# ===============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "⚠️ No se encontró la variable de entorno OPENAI_API_KEY. "
        "Por favor configúrala en Streamlit Cloud (Settings → Secrets)."
    )
    st.stop()
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ===============================
# Título de la app
# ===============================
st.title("📄 Resumen de PDFs con LangChain y OpenAI")

# ===============================
# Carga del PDF
# ===============================
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file:
    try:
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        if not text:
            st.warning("El PDF no contiene texto legible.")
            st.stop()

        # ===============================
        # División de texto
        # ===============================
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)

        # ===============================
        # Crear embeddings y vectorstore
        # ===============================
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # ===============================
        # Configurar QA
        # ===============================
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # ===============================
        # Pregunta del usuario
        # ===============================
        query = st.text_input("Escribe tu pregunta sobre el PDF:")
        if query:
            with st.spinner("🧠 Procesando tu consulta..."):
                answer = qa.run(query)
            st.success("✅ Respuesta:")
            st.write(answer)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {e}")
else:
    st.info("Sube un PDF para empezar a interactuar.")
