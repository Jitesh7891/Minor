import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

# Load datasets (ensure correct paths)
movies = pd.read_csv("./ml-latest-small/ml-latest-small/movies.csv")
ratings = pd.read_csv("./ml-latest-small/ml-latest-small/ratings.csv")

# Preserve original genres and one-hot encode the genres
movies["genre_string"] = movies["genres"]
genre_data = movies["genres"].str.get_dummies(sep='|')
movies = pd.concat([movies, genre_data], axis=1)
movies.drop(columns=["genres"], inplace=True)

# Merge ratings with movies to attach genre info to each rating
ratings_genres = ratings.merge(movies[['movieId'] + list(genre_data.columns)], on='movieId', how='left')

# For each genre, calculate the weighted rating for each user (only for movies that belong to that genre)
for genre in genre_data.columns:
    ratings_genres[genre + '_weighted'] = ratings_genres[genre] * ratings_genres['rating']
    ratings_genres[genre + '_count'] = ratings_genres[genre]

# Aggregate to create a user-genre profile: average rating per genre
user_profiles = {}
for genre in genre_data.columns:
    # Sum weighted ratings and counts per user
    genre_sum = ratings_genres.groupby('userId')[genre + '_weighted'].sum()
    count_sum = ratings_genres.groupby('userId')[genre + '_count'].sum()
    # Compute average rating (handle division by zero)
    user_profiles[genre] = (genre_sum / count_sum).fillna(0)

# Create a DataFrame: rows as users, columns as genres (this serves as our collaborative & content hybrid profile)
user_genre_profile = pd.DataFrame(user_profiles)

# Standardize the aggregated features
scaler = StandardScaler()
profile_scaled = scaler.fit_transform(user_genre_profile)

# Apply PCA to reduce dimensions (using 2 principal components)
pca = PCA(n_components=2)
profile_pca = pca.fit_transform(profile_scaled)

# Create a DataFrame with PCA components for each user
user_pca_df = pd.DataFrame(profile_pca, index=user_genre_profile.index, columns=['PC1', 'PC2'])

# Plot the PCA results for visualization (optional)
plt.figure(figsize=(8, 6))
plt.scatter(user_pca_df['PC1'], user_pca_df['PC2'], alpha=0.6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("User Genre Profile PCA")
# plt.show()

# Select 5 users for recommendations (using a fixed list for reproducibility)
random_users = [1, 2, 3, 4, 5]
print("Selected Users for Recommendation:", random_users)

# Define a helper function for cosine similarity
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

# Provide top 5 hybrid movie recommendations per user
for user in random_users:
    # Get PCA vector for current user and compute distances to all other users
    user_vec = user_pca_df.loc[user].values.reshape(1, -1)
    distances = euclidean_distances(user_vec, user_pca_df)[0]
    distances_series = pd.Series(distances, index=user_pca_df.index)
    
    # Exclude the user itself and find the 5 most similar users (smallest distances)
    distances_series = distances_series.drop(user)
    similar_users = distances_series.nsmallest(5).index
    
    # Get movies that the current user has already rated
    user_rated_movies = set(ratings[ratings['userId'] == user]['movieId'])
    
    # Collaborative component: Compute average ratings for movies by the similar users
    similar_ratings = ratings[ratings['userId'].isin(similar_users)]
    movie_scores = similar_ratings.groupby('movieId')['rating'].mean()
    
    # Exclude movies already rated by the target user
    candidate_movies = movie_scores.drop(user_rated_movies, errors='ignore')
    
    # Content component: Get the user's content profile (their average genre ratings)
    user_content_profile = user_genre_profile.loc[user, list(genre_data.columns)].values.astype(float)
    
    # Compute a hybrid score for each candidate movie: (collaborative score * content similarity)
    hybrid_scores = {}
    for movie_id in candidate_movies.index:
        # Get candidate movie's genre vector from the movies DataFrame
        movie_row = movies[movies['movieId'] == movie_id]
        if movie_row.empty:
            continue
        movie_genre_vector = movie_row[list(genre_data.columns)].values.flatten().astype(float)
        content_sim = cosine_similarity(user_content_profile, movie_genre_vector)
        collab_score = candidate_movies.loc[movie_id]
        hybrid_score = collab_score * content_sim  # You can adjust this formula or weighting if desired
        hybrid_scores[movie_id] = hybrid_score
    
    # Sort candidate movies by hybrid score in descending order and pick top 5
    if len(hybrid_scores) == 0:
        top_recommendations = []
    else:
        sorted_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top_recommendations = [movie_id for movie_id, score in sorted_movies[:5]]
    
    print(f"User {user} (Similar Users: {list(similar_users)}) Top Hybrid Recommendations:")
    print(movies[movies['movieId'].isin(top_recommendations)][['movieId', 'title']].to_string(index=False))
    print("\n")
