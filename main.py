import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from io import StringIO

# === LLM loading ===
def load_LLM(openai_api_key: str):
    return ChatOpenAI(
        temperature=0,
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo"
    )

# === Streamlit page setup ===
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")

# Intro instructions
col1, col2 = st.columns(2)
with col1:
    st.markdown("ChatGPT cannot summarize very long texts. Use this app for long document summarization.")
with col2:
    st.write("Contact [AI Accelera](https://aiaccelera.com) to build AI Projects")

# OpenAI API key input
st.markdown("## Enter Your OpenAI API Key")
openai_api_key = st.text_input(
    label="OpenAI API Key",
    placeholder="Ex: sk-XXXX...",
    type="password"
)

# File uploader
st.markdown("## Upload the text file you want to summarize")
uploaded_file = st.file_uploader("Choose a file", type="txt")

# Output
st.markdown("### Summary")

if uploaded_file:
    file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    
    if len(file_content.split()) > 20000:
        st.warning("File too long. Maximum length is 20,000 words.")
        st.stop()
    
    if not openai_api_key:
        st.warning(
            'Please enter OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)',
            icon="⚠️"
        )
        st.stop()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=5000,
        chunk_overlap=350
    )
    documents = text_splitter.create_documents([file_content])

    # Load LLM and summarization chain
    llm = load_LLM(openai_api_key)
    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce"
    )

    # Run summarization
    summary = summarize_chain.run(documents)
    st.write(summary)
