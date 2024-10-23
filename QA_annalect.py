import os
import re
import pandas as pd
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Demo CSV file URL
demo_csv_url = "https://raw.githubusercontent.com/ggandara13/langchain/refs/heads/main/annalect_website_wata.csv"

# Fetch the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is set
if not openai_api_key:
    st.error("ALERT: OpenAI API key is not set. Please configure it in GitHub Secrets.")

# Initialize OpenAI client with API key
llm = ChatOpenAI(model="gpt-4")

# Function to find the most similar traffic sources using KNN based on aggregated Time on Page
def find_similar_traffic_sources_by_traffic(df, n_similar=3):  # Default to 3
    traffic_source_col = 'Traffic Source'
    time_on_page_col = 'Time on Page'
    
    # Aggregate the 'Time on Page' for each traffic source
    df_aggregated = df.groupby(traffic_source_col).mean().reset_index()

    # Scale the aggregated 'Time on Page' values
    scaler = StandardScaler()
    time_on_page_scaled = scaler.fit_transform(df_aggregated[[time_on_page_col]])

    # Fit the KNN model based on the aggregated 'Time on Page'
    knn = NearestNeighbors(n_neighbors=n_similar + 1)  # +1 to account for the source itself
    knn.fit(time_on_page_scaled)

    distances, indices = knn.kneighbors(time_on_page_scaled)

    # Find the most similar traffic sources, excluding the source itself
    similar_sources = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        source = df_aggregated.iloc[i][traffic_source_col]
        # Exclude the self-match (first neighbor) and get the requested number of most similar sources
        most_similar = df_aggregated.iloc[idx[1:n_similar+1]][traffic_source_col].values
        similarity_scores = dist[1:n_similar+1]  # Exclude the self-match distance
        similar_sources.append({
            'source': source,
            'most_similar': most_similar,
            'similarity_scores': similarity_scores
        })

    return similar_sources

# Function to format the output of KNN results
def format_knn_output(similar_sources, n_similar):
    formatted_output = f"### The {n_similar} most similar traffic sources for each individual traffic source are as follows:\n\n"
    for item in similar_sources:
        source = item['source']
        formatted_output += f"- **{source}** is most similar to:\n"
        for similar, score in zip(item['most_similar'], item['similarity_scores']):
            formatted_output += f"  - **{similar}** with a similarity score of **{score:.4f}**\n"
        formatted_output += "\n"
    return formatted_output

# Function to rank traffic sources based on the total "Time on Page"
def rank_traffic_sources(df):
    # Sum the "Time on Page" for each traffic source and sort by the total
    ranked_df = df.groupby('Traffic Source')['Time on Page'].sum().reset_index()
    ranked_df = ranked_df.sort_values(by='Time on Page', ascending=False)

    # Generate a readable output for the ranking
    ranked_list = [
        f"{i + 1}. {row['Traffic Source']} (Total Time on Page: {row['Time on Page']:.2f})"
        for i, row in ranked_df.iterrows()
    ]

    return "\n".join(ranked_list)

# Global variable to store the DataFrame
df = None

# Define tools/actions for the agent
def knn_tool(question):
    n_similar = extract_number_from_question(question)
    result = find_similar_traffic_sources_by_traffic(df, n_similar)
    return format_knn_output(result, n_similar)

def rank_tool(question):
    return rank_traffic_sources(df)

# Helper function to extract the number of most similar items from the user's question
def extract_number_from_question(question):
    match = re.search(r'(\d+)', question)
    if match:
        return int(match.group(1))
    return 3  # Default to 3 if no number is specified

# Create LangChain tools that the agent can use
tools = [
    Tool(name="KNN Similarity", func=lambda question: knn_tool(question), description="Find similar traffic sources using KNN"),
    Tool(name="Rank Traffic Sources", func=lambda question: rank_tool(question), description="Rank traffic sources by time on page")
]

# Initialize the agent with tools
agent_executor = initialize_agent(
    tools=tools, 
    llm=llm, 
    agent_type="zero-shot-react-description"
)

# Streamlit UI code
st.set_page_config(layout="wide")

# Layout for logo and title
col1, col2 = st.columns([1, 4])

with col1:
    st.image("https://media.licdn.com/dms/image/v2/D4E16AQG2TXgmoHdTOQ/profile-displaybackgroundimage-shrink_200_800/profile-displaybackgroundimage-shrink_200_800/0/1695894586381?e=2147483647&v=beta&t=-OC8NoCyAz_Hh1USOtFfGb7xrxWzicNW00Hx5RH1I6I", width=350)

with col2:
    st.markdown("<h2 style='text-align: left; color: yellow;'>LangChain LLM Agent Demo for Annalect</h2>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col1:
    uploaded_file = st.file_uploader("Upload a CSV file containing 'Traffic Source' and 'Time on Page' - by default a DEMO is loaded", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)  # Store the uploaded data globally
        st.write("Uploaded Data:")
        st.dataframe(df)
    else:
        st.write("No file uploaded. Using demo data.")
        df = pd.read_csv(demo_csv_url)  # Load demo CSV from URL
        st.write("Demo Data:")
        st.dataframe(df)

with col2:
    if df is not None:
        question = st.text_input('Input your question for analyzing the table (e.g., "Give me the 3 most similar traffic sources")')

        if st.button("Perform Analysis"):
            if question:
                # Run the agent executor with the question
                try:
                    # Get the result from the agent
                    result = agent_executor.run(input=question)

                    # Display the result with better formatting
                    st.markdown(f"### Action selected based on the question:\n{question}\n")
                    st.markdown(f"{result}", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Download result as text file
    if 'result' in locals():
        st.download_button(
            label="Download Result",
            data=str(result),
            file_name="analysis_result.txt",
            mime="text/plain"
        )
