import os
import requests
import tempfile

import streamlit as st
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI  # Correct import for ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# Fetch the API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Check if API key is set
if not openai_api_key:
    st.error("ALERT: OpenAI API key is not set. Please configure it in GitHub Secrets.")


def load_and_embed_document(url):
    st.info('Downloading and processing document...')
    response = requests.get(url)
    if response.status_code == 200:
        bytes_data = response.content
        st.success('Document downloaded successfully!.')
    else:
        st.error('Failed to download the document. Please check the URL.')
        return None

    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp.write(bytes_data)
    temp.close()

    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(temp.name)
    documents = loader.load()
    os.unlink(temp.name)  # Ensure the temp file is deleted after loading

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)    

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever()
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)

    return crc

# Set wide mode
st.set_page_config(layout="wide")

# Create three columns: Image, Spacer, Interaction
col1, col2, col3 = st.columns([2, 1, 5])  # Adjusted column widths

# Column 1: Display the image
with col1:
    st.image("https://raw.githubusercontent.com/ggandara13/langchain/main/LLM_book.jpg", use_column_width=True)

# Column 2: Spacer - Empty for spacing
with col2:
    st.write("")

# Column 3: Title and interaction (question and answer)
with col3:
    st.title('Q&A - Large Language Models: A Deep Dive')

    # Load the document once using session state
    if 'crc' not in st.session_state:
        document_url = "https://raw.githubusercontent.com/ggandara13/langchain/main/Large_LanguageModels_A_Deep_Dive_P1.pdf"
        crc = load_and_embed_document(document_url)
        if crc:
            st.session_state.crc = crc
            st.success('Document is ready for questions.')
        else:
            st.error('Document loading failed.')

    # Input for the question
    question = st.text_input('Input your question')

    # Show answer if a question is asked
    if question:
        if 'crc' in st.session_state:
            crc = st.session_state.crc
            response = crc.run({'question': question, 'chat_history': []})  # Pass an empty history

            st.write("Answer:", response)  # Display only the current answer
