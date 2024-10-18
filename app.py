import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_json('/workspaces/Recipe-Ingredients/train.json')
cosine_sim = pd.read_pickle('/workspaces/Recipe-Ingredients/cosine_similarity.pkl') 

# Function to get recommendations based on the recipe index
def get_recommendations(recipe_index, cosine_sim=cosine_sim, top_n=5):
    # Get the pairwise similarity scores for all recipes
    sim_scores = list(enumerate(cosine_sim[recipe_index]))
    
    # Sort the recipes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top N most similar recipes
    sim_scores = sim_scores[1:top_n + 1]  # Exclude the first one (itself)
    
    # Get the recipe indices
    recipe_indices = [i[0] for i in sim_scores]
    
    # Return the top N similar recipes
    return df.iloc[recipe_indices]

# Streamlit app
st.title('Recipe Recommendation System')

# Select a recipe ID
selected_id = st.selectbox('Select a Recipe ID to get recommendations', df['id'].unique())

# Find the index of the selected recipe
selected_recipe = df[df['id'] == selected_id].iloc[0]
selected_index = df[df['id'] == selected_id].index[0]

# Display the cuisine and ingredients of the selected recipe
st.write(f"**Selected Recipe ID**: {selected_id}")
st.write(f"**Cuisine**: {selected_recipe['cuisine']}")
st.write(f"**Ingredients**: {', '.join(selected_recipe['ingredients'])}")

# Get top 5 recommendations for the selected recipe
st.write("## Recommended Recipes:")
recommendations = get_recommendations(selected_index)

# Display recommendations in separate boxes
for i in range(len(recommendations)):
    recommended_recipe = recommendations.iloc[i]
    
    # Create a box for each recommendation
    st.markdown("### Recommendation " + str(i + 1))
    st.markdown(f"**Cuisine**: {recommended_recipe['cuisine']}")
    st.markdown(f"**Ingredients**:")
    for ingredient in recommended_recipe['ingredients']:
        st.markdown(f"- {ingredient}")
    
    # Add a horizontal line for separation
    st.markdown("---")  # This adds a horizontal line