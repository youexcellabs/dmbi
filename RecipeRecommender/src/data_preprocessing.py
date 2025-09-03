import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text):
    """
    Cleans and preprocesses text data (ingredients).
    Removes punctuation, converts to lowercase, removes stop words.
    """
    if pd.isna(text):
        return ""
    # Remove punctuation and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize words
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_recipes(df):
    """
    Applies text cleaning to the 'ingredients' column.
    """
    df['cleaned_ingredients'] = df['ingredients'].apply(clean_text)
    return df

if __name__ == '__main__':
    # Example usage:
    # This block runs only when data_preprocessing.py is executed directly
    # and demonstrates its functionality.
    data = {
        'recipe_id': [1, 2],
        'name': ['Test Recipe 1', 'Test Recipe 2'],
        'ingredients': ['Flour, Sugar, Eggs, Milk, Salt', 'Chicken, Rice, Onion, Garlic, Spices'],
        'cuisine': ['Dessert', 'Asian']
    }
    test_df = pd.DataFrame(data)
    processed_df = preprocess_recipes(test_df.copy())
    print("Original Ingredients:")
    print(test_df['ingredients'])
    print("\nCleaned Ingredients:")
    print(processed_df['cleaned_ingredients'])