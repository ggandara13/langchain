import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import os
import openai  # Import OpenAI for direct API usage

import os
import streamlit as st

# Fetch the API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is set
if openai_api_key:
    st.success(f"OpenAI API key found: {openai_api_key[:5]}...")  # Don't print the whole key for security reasons
else:
    st.error("ALERT: OpenAI API key is not set. Please configure it in GitHub Secrets.")



# URL of the banner image from Kizen
banner_url = "https://kizen.com/dist/images/sharelink-kizenhome.webp"

# List of URLs
urls = [
    'https://kizen.com/content/news/rise-of-chatgpt-could-be-a-boon-for-savvy-business-owners/',
    'https://kizen.com/content/news/most-higher-income-workers-report-interacting-with-ai-in-their-jobs-survey-says/',
    'https://kizen.com/content/news/return-to-office-might-be-making-workers-miserable/',
    'https://kizen.com/content/news/2023-kizen-survey-results/',
    'https://kizen.com/content/news/what-can-small-business-owners-learn-from-amazon/',
    'https://kizen.com/content/news/title-inflation-is-hurting-employee-career-growth-and-company-morale/',
    'https://kizen.com/content/news/salestechstar-interview-with-john-winner-ceo-at-kizen/',
    'https://kizen.com/content/news/with-likely-recession-encroaching-proptech-seen-as-cost-cutting-tool-in-cre/',
    'https://kizen.com/content/news/software-robots-are-gaining-ground-in-white-collar-office-world/',
    'https://kizen.com/insurance/insurance-agent-crm-ai',
    'https://kizen.com/'
]

# Step 1: Load the Articles and Split into Chunks
def load_articles(urls):
    documents = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()
        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        documents.extend(chunks)
    return documents

# Step 2: Compute Embeddings and Step 3: Store Embeddings
def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Step 4: Perform Semantic Search and Step 5: Summarize Results
def semantic_search_and_summarize(query, vectorstore):
    # Use GPT-3.5-turbo-16k model with higher context limit
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k')
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Use "stuff" for simple summarization
        retriever=retriever,
        return_source_documents=True
    )
    # Perform the query
    result = qa_chain({"query": query})
    return result


# Streamlit Interface

# Display Kizen banner
st.image(banner_url, caption="Kizen - The Leader in ProGen AI", use_column_width=True)

# Title of the app
st.title("Article Search and Summarization")

# Input for user query
query = st.text_input("Enter your query", "")

# HTML for the Kizen logo banner
kizen_logo_html = """
<a href="https://kizen.com/" class="logo">
    <img class="logo-white" src="https://kizen.com/dist/images/svg/logo-white.svg" width="158" height="40" alt="Kizen Logo">
</a>
"""

# Display Kizen logo banner using custom HTML
st.markdown(kizen_logo_html, unsafe_allow_html=True)

# If the query is entered, perform the search and summarization
if query:
    st.info("Loading articles...")
    
    # Load articles and create vector store
    documents = load_articles(urls)
    vectorstore = create_vector_store(documents)
    
    st.info("Performing semantic search and summarization...")
    
    # Perform search and summarization
    result = semantic_search_and_summarize(query, vectorstore)
    
    # Display results in Streamlit
    st.header("Summary of Relevant Articles")
    st.write(result['result'])
    
    st.header("Source URLs")
    for doc in result['source_documents']:
        st.write(doc.metadata.get('source', 'Unknown'))
