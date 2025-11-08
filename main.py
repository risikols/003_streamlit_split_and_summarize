import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from io import StringIO

# LLM loader
def load_LLM(openai_api_key):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm

# Page setup
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")

# Intro
col1, col2 = st.columns(2)
with col1:
    st.markdown("ChatGPT cannot summarize long texts. Now you can do it with this app.")
with col2:
    st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")

# OpenAI API Key
st.markdown("## Enter Your OpenAI API Key")
openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="Ex: sk-xxxx")

# File uploader
st.markdown("## Upload the text file you want to summarize")
uploaded_file = st.file_uploader("Choose a file", type="txt")

st.markdown("### Here is your Summary:")

if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    file_input = stringio.read()

    if len(file_input.split(" ")) > 20000:
        st.write("Please enter a shorter file. Maximum 20000 words.")
        st.stop()

    if not openai_api_key:
        st.warning(
            'Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️"
        )
        st.stop()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=5000,
        chunk_overlap=350
    )
    splitted_documents = text_splitter.create_documents([file_input])

    # Load LLM
    llm = load_LLM(openai_api_key)

    # Summarize
    summarize_chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
    summary_output = summarize_chain.run(splitted_documents)

    st.write(summary_output)

