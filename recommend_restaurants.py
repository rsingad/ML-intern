import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load dataset
data_path = "Dataset (1).csv"
df = pd.read_csv(data_path, encoding='latin-1')

# Data Preprocessing
# Handling missing values in Cuisines
df['Cuisines'] = df['Cuisines'].fillna('Unknown')

# Feature Engineering for Recommendation
# 1. TF-IDF on Cuisines to capture similarity in food types
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Cuisines'])

# 2. Numerical Features: Price range and Aggregate rating
scaler = MinMaxScaler()
numerical_features = scaler.fit_transform(df[['Price range', 'Aggregate rating']])

# 3. Combine TF-IDF and Numerical Features
# We give some weight to Cuisines and some to Price/Rating
combined_features = np.hstack([tfidf_matrix.toarray(), numerical_features])

# Compute Cosine Similarity (Using a subset if memory is an issue, but 9500 rows should be fine)
print("Computing similarity matrix...")
cosine_sim = cosine_similarity(combined_features, combined_features)

def get_recommendations(restaurant_name, top_n=5):
    try:
        # Get index of the restaurant
        idx = df[df['Restaurant Name'].str.lower() == restaurant_name.lower()].index[0]
        
        # Get similarity scores for all restaurants with this one
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort by similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar restaurants (excluding itself)
        sim_scores = sim_scores[1:top_n+1]
        
        # Get indices
        restaurant_indices = [i[0] for i in sim_scores]
        
        return df.iloc[restaurant_indices][['Restaurant Name', 'Cuisines', 'Price range', 'Aggregate rating']]
    except IndexError:
        return f"Restaurant '{restaurant_name}' not found in the dataset."

# Test the system
print("\nTesting Recommendations for 'Ooma':")
recommendations = get_recommendations('Ooma')
print(recommendations)

print("\nTesting Recommendations for 'Starbucks':")
recommendations = get_recommendations('Starbucks')
if isinstance(recommendations, pd.DataFrame):
     print(recommendations)
else:
     print(recommendations)

# Example: User preference based recommendation (using a theoretical ideal)
def recommend_by_preference(preferred_cuisine, price_range, top_n=5):
    # Vectorize the preferred cuisine
    pref_tfidf = tfidf.transform([preferred_cuisine]).toarray()
    
    # Scale the preferred price range and a high rating (e.g., 5.0)
    pref_num = scaler.transform([[price_range, 5.0]])
    
    # Combine
    pref_combined = np.hstack([pref_tfidf, pref_num])
    
    # Compute similarity between preference and all restaurants
    sim_scores = list(enumerate(cosine_similarity(pref_combined, combined_features)[0]))
    
    # Sort and get top N
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n]
    
    restaurant_indices = [i[0] for i in sim_scores]
    return df.iloc[restaurant_indices][['Restaurant Name', 'Cuisines', 'Price range', 'Aggregate rating']]

print("\nTesting Recommendations for preference: Cuisine='Italian', Price Range=3")
pref_recommendations = recommend_by_preference('Italian', 3)
print(pref_recommendations)
