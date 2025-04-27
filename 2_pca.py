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

# Create a DataFrame: rows as users, columns as genres
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

# Select 5 random users for recommendations
# random_users = np.random.choice(user_genre_profile.index, 5, replace=False)
random_users=[1,2,3,4,5]
print("Selected Users for Recommendation:", random_users)

# Provide top 5 movie recommendations per user based on similar users in PCA space
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
    
    # Compute average ratings for movies by the similar users
    similar_ratings = ratings[ratings['userId'].isin(similar_users)]
    movie_scores = similar_ratings.groupby('movieId')['rating'].mean()
    
    # Exclude movies already rated by the current user
    recommended_movies = movie_scores.drop(user_rated_movies, errors='ignore')
    top_recommendations = recommended_movies.nlargest(5).index
    
    print(f"User {user} (Similar Users: {list(similar_users)}) Top Recommendations:")
    print(movies[movies['movieId'].isin(top_recommendations)][['movieId', 'title']].to_string(index=False))
    print("\n")
