import os
import tempfile
import streamlit as st
import cloudinary
import cloudinary.uploader
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Configuration for Cloudinary
cloudinary.config(
    cloud_name="dmh9uua2k",
    api_key="684786698553791",
    api_secret="z5Y_OKnI6yUkxlL19d79eakfz88",
    secure=True
)

# Fetch the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is set
if not openai_api_key:
    st.error("ALERT: OpenAI API key is not set. Please configure it in GitHub Secrets.")

# Initialize OpenAI client with API key
client = OpenAI(api_key=openai_api_key)

# Upload the image to Cloudinary and get a public URL
def upload_image_to_cloudinary(image_path):
    upload_result = cloudinary.uploader.upload(image_path)
    return upload_result.get("secure_url", None)

# Define the function to describe the image
def describe_image(img_url):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", 
                     "text": '''
The context of the image is a product from Saks Fifth Avenue retail store, provide the detail description of the product
                       '''},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url,
                        },
                    },
                ],
            }
        ],
        max_tokens=500,
    )
    content = response.choices[0].message.content
    return content

# Load the CSV file and remove duplicates based on 'name' and 'description'
def load_csv_data():
    url = "https://raw.githubusercontent.com/ggandara13/langchain/refs/heads/main/adidas_usa.csv"
    df = pd.read_csv(url)

    # Remove duplicates based on 'name' and 'description'
    df.drop_duplicates(subset=['name', 'description'], inplace=True)

    # Calculate the total number of rows after removing duplicates
    total_rows = df.shape[0]
    
    return df, total_rows

# Function to compute embeddings for descriptions using the transformer model
def get_sentence_embeddings(model, tokenizer, descriptions):
    inputs = tokenizer(descriptions.tolist(), padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Streamlit UI code
st.set_page_config(layout="wide")

# Insert the logo and title in the top row with two columns
top_col1, top_col2 = st.columns([1, 1.5])
with top_col1:
    logo_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT4fxXkyDGBQ3Dc9W9QK9mevDEQX5BFEUXTdw&s"
    st.image(logo_url, width=500)  # Adjust the width to fit your logo size

with top_col2:
    st.markdown("""
    <h1 style='font-size:24px;color: yellow;'>Image Uploader - Analyzer AI : Saks Demo</h1>

    <p style='font-size:18px;'>Upload an image to find similarity products</p>
    """, unsafe_allow_html=True)

# Create a two-column layout for image upload and analysis
col1, col2 = st.columns([1, 1.5])

# Load the transformer model and tokenizer
m_mpnet = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Left Column: Image upload and display
with col1:
    uploaded_file = st.file_uploader("Upload an image (JPG)", type=["jpg", "jpeg"])

    if uploaded_file:
        # Save the uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_file.write(uploaded_file.read())
        temp_file.close()

        # Display the uploaded image (resized to fit vertically)
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

# Right Column: Button to get description and display result
with col2:
    if uploaded_file and st.button("Perform Analysis IMAGE Description"):
        # Upload the image to Cloudinary and get a public URL
        image_url = upload_image_to_cloudinary(temp_file.name)

        if image_url:
            # Process the image using the `describe_image` function
            description = describe_image(image_url)

            # Save the description to session state to retain after button clicks
            st.session_state['description'] = description

        # Clean up the temporary file after use
        os.unlink(temp_file.name)

    # Display the description from session state (only if it exists)
    if 'description' in st.session_state:
        description = st.session_state['description']
        st.markdown(f"**Image Description:**\n\n{description}")
        st.download_button(
            label="Download as Text",
            data=description,
            file_name="description.txt",
            mime="text/plain",
            key="download_button"
        )

# Load the CSV Data and remove duplicates, get total rows
df, total_rows = load_csv_data()

# Create a button to find similar products
if 'description' in st.session_state:
    if st.button("Find the Top 10 Similar Products"):
        # Get embeddings for the image description
        img_description_embedding = get_sentence_embeddings(m_mpnet, tokenizer, pd.Series([st.session_state['description']]))

        # Get embeddings for the product descriptions in the CSV
        csv_description_embeddings = get_sentence_embeddings(m_mpnet, tokenizer, df['description'])

        # Calculate cosine similarities
        similarities = cosine_similarity(img_description_embedding, csv_description_embeddings)

        # Add similarities to the DataFrame and display the top 10 results
        df['similarity'] = similarities[0]
        
        # Select the required columns for output: name, sku, description, similarity
        df_output = df[['name', 'sku', 'description', 'similarity']].sort_values(by='similarity', ascending=False).head(10)

        # Display the similarity result in the second column (col2) below the image description
        with col2:
            st.markdown("### Top 10 Similar Products")
            st.dataframe(df_output)
            
        # Display the total rows processed after the similarity results
        with col2:
            st.markdown(f"<small>Total of rows processed: datasource {total_rows}</small>", unsafe_allow_html=True)

# Outside of the columns, display the full dataframe used for finding similarities
st.markdown("### DataFrame that will be used for Similarity Calculation")
st.dataframe(df)