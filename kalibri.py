import os
import tempfile
import streamlit as st
import cloudinary
import cloudinary.uploader
from openai import OpenAI

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
The context of the image is data related to hotels, can be demand, revenue, stays, analyze the data providing insights and identifying best recommendations
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

# Streamlit UI code
st.set_page_config(layout="wide")

# Inject CSS for background and text colors
st.markdown(
    """
    <style>
    body {
        color: #000000;
        background-color: #F0F0F0;
    }
    h1 {
        color: #FFD700;
    }
    </style>
    """, unsafe_allow_html=True)

# Insert the logo and title in the top row with two columns
top_col1, top_col2 = st.columns([1, 4])
with top_col1:
    logo_url = "https://www.kalibrilabs.com/hs-fs/hubfs/Kalibri_Logo_Horizontal-White_Tagline.webp?width=400&height=108&name=Kalibri_Logo_Horizontal-White_Tagline.webp"
    st.image(logo_url, width=300)  # Adjust the width to fit your logo size

with top_col2:
    st.markdown("""
    <h1 style='font-size:24px;'>Image Uploader - Analyzer AI : KALIBRI Labs</h1>
    <p style='font-size:18px;'>Upload an image to analyze data related to hotels</p>
    """, unsafe_allow_html=True)

# Create a two-column layout for image upload and analysis
col1, col2 = st.columns([1.5, 1])

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
    if uploaded_file and st.button("Perform Analysis"):
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
            label="Dowload as Text",
            data=description,
            file_name="description.txt",
            mime="text/plain",
            key="download_button"
        )
