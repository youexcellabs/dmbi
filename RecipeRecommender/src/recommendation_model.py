import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib # To save/load the model components
import os

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.recipes_df = None
        self.recipe_id_to_index = {}
        self.index_to_recipe_id = {}

    def fit(self, recipes_df_processed):
        """
        Trains the TF-IDF vectorizer and computes the cosine similarity matrix.
        Assumes recipes_df_processed has a 'cleaned_ingredients' column.
        """
        self.recipes_df = recipes_df_processed.copy()
        
        # Create mappings between recipe_id and internal DataFrame index
        self.recipe_id_to_index = {recipe_id: index for index, recipe_id in enumerate(self.recipes_df['recipe_id'])}
        self.index_to_recipe_id = {index: recipe_id for index, recipe_id in enumerate(self.recipes_df['recipe_id'])}

        # Fit TF-IDF Vectorizer on cleaned ingredients
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.recipes_df['cleaned_ingredients'])
        print(f"TF-IDF Matrix shape: {self.tfidf_matrix.shape}")

    def recommend(self, recipe_id, num_recommendations=5):
        """
        Recommends similar recipes based on a given recipe_id.
        """
        if recipe_id not in self.recipe_id_to_index:
            print(f"Error: Recipe ID '{recipe_id}' not found in the dataset.")
            return pd.DataFrame() # Return empty DataFrame

        idx = self.recipe_id_to_index[recipe_id]

        # Calculate cosine similarity with all other recipes
        cosine_similarities = cosine_similarity(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()

        # Get the indices of the most similar recipes, excluding the recipe itself
        # argsort sorts in ascending order, so [:-num_recommendations-1:-1] gets top N
        similar_recipe_indices = cosine_similarities.argsort()[:-num_recommendations-2:-1] 
        
        # Convert indices back to original recipe_ids and filter out the input recipe
        recommended_recipe_ids = [self.index_to_recipe_id[i] for i in similar_recipe_indices if self.index_to_recipe_id[i] != recipe_id]

        # Get details of recommended recipes
        recommended_recipes = self.recipes_df[self.recipes_df['recipe_id'].isin(recommended_recipe_ids)]
        
        # Sort recommendations by similarity (optional, but good for display)
        # Note: Re-calculating or storing similarity scores for precise sorting here would be more complex
        # For simplicity, we'll just return the top N by index.
        
        return recommended_recipes[['recipe_id', 'name', 'ingredients', 'cuisine', 'prep_time_minutes']]

    def save_model(self, path="models/content_based_model.pkl"):
        """Saves the trained TF-IDF vectorizer, TF-IDF matrix, and mappings."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_components = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'recipes_df': self.recipes_df, # Store the processed DataFrame as well
            'recipe_id_to_index': self.recipe_id_to_index,
            'index_to_recipe_id': self.index_to_recipe_id
        }
        joblib.dump(model_components, path)
        print(f"Model saved to {path}")

    def load_model(self, path="models/content_based_model.pkl"):
        """Loads the trained model components."""
        if not os.path.exists(path):
            print(f"Error: Model file not found at {path}")
            return False
        model_components = joblib.load(path)
        self.tfidf_vectorizer = model_components['tfidf_vectorizer']
        self.tfidf_matrix = model_components['tfidf_matrix']
        self.recipes_df = model_components['recipes_df']
        self.recipe_id_to_index = model_components['recipe_id_to_index']
        self.index_to_recipe_id = model_components['index_to_recipe_id']
        print(f"Model loaded from {path}")
        return True

if __name__ == '__main__':
    # This block demonstrates how to use the recommender when run directly.
    # It mimics what you'd do in a Jupyter Notebook or a main script.
    
    # Ensure data/recipes.csv exists and src/data_preprocessing.py is correct
    from data_preprocessing import preprocess_recipes

    # Load data
    try:
        df = pd.read_csv('../data/recipes.csv')
    except FileNotFoundError:
        print("Error: recipes.csv not found. Make sure it's in the 'data/' directory.")
        exit()

    # Preprocess data
    processed_df = preprocess_recipes(df.copy())

    # Initialize and fit the recommender
    recommender = ContentBasedRecommender()
    recommender.fit(processed_df)

    # Save the model
    recommender.save_model(path='../models/content_based_model.pkl')

    # Load the model (demonstrating loading)
    loaded_recommender = ContentBasedRecommender()
    if loaded_recommender.load_model(path='../models/content_based_model.pkl'):
        # Get recommendations for a recipe (e.g., Spaghetti Carbonara, recipe_id=1)
        print("\nRecommendations for 'Spaghetti Carbonara' (recipe_id=1):")
        recommendations = loaded_recommender.recommend(recipe_id=1, num_recommendations=3)
        print(recommendations)

        print("\nRecommendations for 'Chicken Tikka Masala' (recipe_id=2):")
        recommendations = loaded_recommender.recommend(recipe_id=2, num_recommendations=2)
        print(recommendations)

        print("\nRecommendations for 'Sushi Rolls' (recipe_id=8):")
        recommendations = loaded_recommender.recommend(recipe_id=8, num_recommendations=3)
        print(recommendations)

        print("\nRecommendations for an unknown recipe (recipe_id=99):")
        recommendations = loaded_recommender.recommend(recipe_id=99, num_recommendations=3)
        print(recommendations)