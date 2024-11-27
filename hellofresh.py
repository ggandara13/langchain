import streamlit as st
import pandas as pd
import os
import torch
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
import openai
import re
from PIL import Image
import requests
from io import BytesIO
from sentence_transformers import util
import torch
from transformers import AutoTokenizer, AutoModel


# Set page configuration for wide layout
st.set_page_config(layout="wide")

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the MPNet model and tokenizer
mpnet_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
mpnet_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# Load pickle files
@st.cache_data
def load_data(max_rows=10000):
    try:
        pickle_files = sorted([os.path.join("split_pickles", file) for file in os.listdir("split_pickles") if file.endswith('.pkl')])
        df_combined = pd.concat([pd.read_pickle(file) for file in pickle_files])
        df_filtered = df_combined[df_combined['calories'] <= 2000]
        
        # Create a visual subset for display purposes
        selected_columns = [
            "title", "ingredients", "categories", "calories",
            "sodium", "fat", "protein", "rating", "prep_time"
        ]
        df_visual = df_filtered[selected_columns].head(max_rows)
        return df_filtered.head(max_rows), df_visual
    except Exception as e:
        st.error(f"Error loading pickle files: {e}")
        return None, None

# Load the HelloFresh logo
@st.cache_data
def load_logo():
    logo_url = "https://cdn.freebiesupply.com/logos/large/2x/hellofresh-logo.png"
    response = requests.get(logo_url)
    return Image.open(BytesIO(response.content))

def load_logo2():
    logo_url = "https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiapSQHY0zKxjil7et9AgyxJ2mPWjE6MqCHrdCC-TQTDT2UhBKvseAZ4ytDzmvFEyZ44Exx1wSHcowFsZyJGBu6IOXhenxaxSNGOq8COU5i1dtOjrZQitAycX6ERUwylIi4KvJyLpogGgas/s1600/Hello+Fresh+%25283%2529.jpg"
    response = requests.get(logo_url)
    return Image.open(BytesIO(response.content))


# Recipe constraints parsing
def extract_recipe_and_constraints_openai_v2(query):
    """
    Parse the query to extract the main recipe and constraints using OpenAI API.
    """
    prompt = f"""
    Parse the following query into a main recipe and constraints. Constraints include calories, sodium, protein, fat, and prep_time.

    Query: "{query}"

    Output format:
    Recipe: <recipe>
    Constraints: calories < x, sodium < y, protein > z, fat < a, prep_time < b
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0,
        )
        result = response.choices[0].message.content

        # Extract the recipe and constraints
        recipe_match = re.search(r"Recipe:\s*(.*)", result)
        constraints_match = re.search(r"Constraints:\s*(.*)", result)

        recipe = recipe_match.group(1).strip() if recipe_match else None
        constraints_str = constraints_match.group(1).strip() if constraints_match else ""

        # Parse constraints into a dictionary
        constraints = parse_constraints_from_openai_output(constraints_str)

        return recipe, constraints

    except Exception as e:
        st.error(f"Error during OpenAI API call: {e}")
        return None, {}

import re

def parse_constraints_from_openai_output(constraints_str):
    """
    Parses constraints from the OpenAI output string into a dictionary.
    Sets unspecified constraints to None, so they are ignored in filtering.

    Args:
        constraints_str (str): Constraints string from OpenAI output.

    Returns:
        dict: Parsed constraints with numerical values or None for unspecified constraints.
    """
    constraints = {
        'calories': None,
        'sodium': None,
        'protein': None,
        'fat': None,
        'prep_time': None
    }
    
    print("----constraints_str:", constraints_str)

    # Detect calorie constraint
    calorie_match = re.search(r"(calories|calorie)\s*[<≤]\s*(\d+)", constraints_str, re.IGNORECASE)
    if calorie_match:
        constraints['calories'] = int(calorie_match.group(2))

    # Detect sodium constraint
    sodium_match = re.search(r"(sodium|salt)\s*[<≤]\s*(\d+)", constraints_str, re.IGNORECASE)
    if sodium_match:
        constraints['sodium'] = int(sodium_match.group(2))
    elif re.search(r"(low sodium|low-sodium|less sodium)", constraints_str, re.IGNORECASE):
        constraints['sodium'] = 135  # Default threshold for "low sodium"

    # Detect protein constraint
    protein_match = re.search(r"(protein|high protein)\s*[>≥]\s*(\d+)", constraints_str, re.IGNORECASE)
    if protein_match:
        constraints['protein'] = int(protein_match.group(2))
    elif re.search(r"(high protein|protein > high)", constraints_str, re.IGNORECASE):
        constraints['protein'] = 10  # Default threshold for "high protein"

    # Detect fat constraint - do not set a default if unspecified
    fat_match = re.search(r"(fat)\s*[<≤]\s*(\d+)", constraints_str, re.IGNORECASE)
    if fat_match:
        constraints['fat'] = int(fat_match.group(2))

    # Detect prep_time constraint
    prep_time_match = re.search(r"(prep_time|prep time|preparation time|under|less than)\s*[<≤]?\s*(\d+)\s*(min|mins|minutes)?", constraints_str, re.IGNORECASE)
    if prep_time_match:
        constraints['prep_time'] = int(prep_time_match.group(2))
    
    return constraints

def filter_recipes_by_constraints(df_recipes, constraints):
    # Filter based on each constraint if it is not None
    filtered_recipes = df_recipes
    if constraints['calories'] is not None:
        filtered_recipes = filtered_recipes[filtered_recipes['calories'] <= constraints['calories']]
    if constraints['sodium'] is not None:
        filtered_recipes = filtered_recipes[filtered_recipes['sodium'] <= constraints['sodium']]
    if constraints['protein'] is not None:
        filtered_recipes = filtered_recipes[filtered_recipes['protein'] >= constraints['protein']]
    if constraints['fat'] is not None:
        filtered_recipes = filtered_recipes[filtered_recipes['fat'] <= constraints['fat']]
    if constraints['prep_time'] is not None:
        filtered_recipes = filtered_recipes[filtered_recipes['prep_time'] <= constraints['prep_time']]
    
    return filtered_recipes


# Load MPNet model and tokenizer
mpnet_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
mpnet_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

def get_embedding(text):
    # Ensure the input is a string
    if not isinstance(text, str):
        text = str(text)
    inputs = mpnet_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = mpnet_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings



def parse_query_with_embeddings(query, df_recipes, mpnet_model, mpnet_tokenizer):
    """
    Find the most similar recipe to the user's query based on embeddings.

    Args:
        query (str): User's query.
        df_recipes (DataFrame): DataFrame containing recipes with title embeddings.
        mpnet_model: Pre-trained MPNet model for embeddings.
        mpnet_tokenizer: Tokenizer for the MPNet model.

    Returns:
        main_recipe (str): Title of the most similar recipe.
    """
    # Generate embedding for the query using the get_embedding function
    query_embedding = get_embedding(query).squeeze()
    
    # Convert title embeddings in df_recipes to torch tensors for cosine similarity calculation
    title_embeddings_tensor = torch.stack([torch.tensor(embedding) for embedding in df_recipes['title_embedding'].values])
    
    # Calculate cosine similarity between query embedding and recipe title embeddings
    cosine_scores = util.cos_sim(query_embedding, title_embeddings_tensor).squeeze()
    
    # Find the index of the most similar recipe by the highest cosine similarity score
    best_match_idx = cosine_scores.argmax().item()
    main_recipe = df_recipes.iloc[best_match_idx]['title']

    print(f"Best match found: {main_recipe} (Cosine similarity: {cosine_scores[best_match_idx]:.4f})")
    
    return main_recipe



# Main function for recommendations
def recommend_similar_recipes_with_constraints(query, df_recipes, top_n=10):
    # Parse the query to extract the main recipe name and constraints
    user_recipe, constraints = extract_recipe_and_constraints_openai_v2(query)
    print("**** User Recipe *****:", user_recipe)
    print("**** Constraints *****:", constraints)



    # Parse the user recipe to find the most similar recipe in the 20K dataset
    main_recipe = parse_query_with_embeddings(user_recipe, df_recipes, mpnet_model, mpnet_tokenizer)
    
    print("**** Similar Recipe Name **********:", main_recipe)

    # Filter recipes based on parsed constraints
    filtered_recipes = filter_recipes_by_constraints(df_recipes, constraints)
    print("----------> filtered_recipes:SHAPE", filtered_recipes.shape)
    
    if filtered_recipes.empty:
        print("No recipes match the given constraints. Relaxing constraints...")
        filtered_recipes = df_recipes.copy()  # Use unfiltered dataset
        legend = "No recipes matched the constraints. Recommendations are based on similarity to the main recipe without constraints."
    else:
        legend = "Recommendations are based on similarity to the main recipe with the given constraints."
    
    # Check if the main recipe exists in the filtered recipes or the full dataset
    if main_recipe not in filtered_recipes['title'].values:
        if main_recipe in df_recipes['title'].values:
            # Use unfiltered dataset to locate the main recipe embeddings
            main_recipe_embeddings = df_recipes[df_recipes['title'] == main_recipe][
                ['title_embedding', 'ingredients_embedding', 'directions_embedding']
            ].iloc[0]
            legend += f" The main recipe '{main_recipe}' was not found in the filtered results but is considered for similarity in the unfiltered dataset."
        else:
            print(f"The main recipe '{main_recipe}' does not exist in the dataset. Recommendations cannot proceed.")
            return None, f"The main recipe '{main_recipe}' was not found in the dataset."
    else:
        # Retrieve main recipe embeddings from the filtered recipes
        main_recipe_embeddings = filtered_recipes[filtered_recipes['title'] == main_recipe][
            ['title_embedding', 'ingredients_embedding', 'directions_embedding']
        ].iloc[0]

    # Calculate cosine similarity across embeddings for all three fields
    all_embeddings = {
        'title': torch.stack([torch.tensor(embedding) for embedding in filtered_recipes['title_embedding'].values]),
        'ingredients': torch.stack([torch.tensor(embedding) for embedding in filtered_recipes['ingredients_embedding'].values]),
        'directions': torch.stack([torch.tensor(embedding) for embedding in filtered_recipes['directions_embedding'].values])
    }
    
    cosine_scores = (
        0.5 * util.cos_sim(torch.tensor(main_recipe_embeddings['title_embedding']), all_embeddings['title']) +
        0.4 * util.cos_sim(torch.tensor(main_recipe_embeddings['ingredients_embedding']), all_embeddings['ingredients']) +
        0.1 * util.cos_sim(torch.tensor(main_recipe_embeddings['directions_embedding']), all_embeddings['directions'])
    ).squeeze()
    
    # Add similarity scores and sort by score
    filtered_recipes = filtered_recipes.copy()
    filtered_recipes['similarity_score'] = cosine_scores
    recommended_recipes = (
        filtered_recipes[filtered_recipes['title'] != main_recipe]
        .sort_values(by='similarity_score', ascending=False)
        .head(top_n)
    )
        
    # Return recommendations with similarity scores included
    return recommended_recipes[['title', 'calories', 'sodium', 'protein', 'fat', 
                                'prep_time', 'similarity_score']], legend, user_recipe, constraints



import openai

def generate_recipe_from_recommendations(top_recommendations, df_recipes):
    """
    Generate a new recipe using the top recommendations as inspiration.

    Args:
        top_recommendations (pd.DataFrame): DataFrame containing the top 10 recommendations.
        df_recipes (pd.DataFrame): Full recipe dataset with detailed columns like ingredients and directions.

    Returns:
        str: The generated recipe.
    """
    # Extract context from the top recommendations
    context = []
    for idx, row in top_recommendations.iterrows():
        recipe_data = df_recipes.loc[idx]  # Find the full details using the index
        context.append(
            f"Title: {recipe_data['title']}\n"
            f"Ingredients: {', '.join(recipe_data['ingredients'])}\n"
            f"Instructions: {recipe_data['directions_with_notes']}\n"
        )

    # Combine the context
    inspiration_text = "\n\n".join(context)
    prompt = f"""
    Based on the following recipes, create a unique recipe:
    
    {inspiration_text}
    
    Your new recipe should include:
    - A title.
    - A list of ingredients.
    - Step-by-step cooking instructions.
    be sure to delimit each section within brackets as [Title], [Ingredientes], [Directions] to identify each section in the output
    """

    # Generate the recipe using GPT
    response = openai.chat.completions.create(
        #model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful recipe assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    # Extract the generated recipe
    generated_recipe = response.choices[0].message.content
    return generated_recipe



# Initialize session state
if "query_executed" not in st.session_state:
    st.session_state["query_executed"] = False


# Streamlit App
st.title("HelloFresh : Recipe Recommender")

# Create two columns
col1, col2 = st.columns([1, 2])

# Load data
df_recipes, df_visual = load_data(max_rows=10000)

# Column 1: Display HelloFresh logo and GenAI Button
with col1:
    st.image(load_logo(), caption="", use_column_width=True)
    st.image(load_logo2(), caption="HelloFresh - a Prototype by Gerardo Gandara", use_column_width=True)
    
    

# Column 2: Handle user query and recommendations
with col2:
    query = st.text_input("Enter your recipe query:")

    if query:
        # Run recommendations
        top_recommendations, legend, recipe, constraints = recommend_similar_recipes_with_constraints(query, df_recipes)

        # Store that the query has been executed
        st.session_state["query_executed"] = True

        # Display Recipe and Constraints
        st.markdown("### Extracted Recipe and Constraints")
        st.text(f"Recipe: {recipe}")
        st.text(f"Constraints: {constraints}")

        st.markdown("<h4 style='color:yellow;'>Top Recommendations:</h4>", unsafe_allow_html=True)
        st.dataframe(top_recommendations.reset_index(drop=True))


        if st.button("Generate GenAI Recipe"):
            st.markdown("<h4 style='color:yellow;'>Generated Recipe:</h4>", unsafe_allow_html=True)
            st.text("Generated recipe will be displayed here.")
            
            generated_recipe = generate_recipe_from_recommendations(top_recommendations, df_recipes)
            st.text(generated_recipe)

