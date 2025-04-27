import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score

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

# Optional: Standardize the aggregated features
scaler = StandardScaler()
profile_scaled = scaler.fit_transform(user_genre_profile)

# Determine optimal K using Elbow and Silhouette methods
inertia = []
silhouette_scores = []
K_range = range(4, 10)  # Testing K from 2 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(profile_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(profile_scaled, kmeans.labels_))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker="o", color="red")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Method")
# plt.show()

ratings['rating'].hist(bins=20)
plt.xlabel("Rating")
plt.ylabel("Count")
plt.title("Distribution of Ratings in Dataset")
plt.show()


# Select best K based on silhouette score
best_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal K: {best_k}")

# Apply K-Means with the optimal K
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(profile_scaled)
user_genre_profile['Cluster'] = clusters

# Display the number of users in each cluster
cluster_counts = user_genre_profile['Cluster'].value_counts().sort_index()
print("Users per cluster:")
print(cluster_counts)

# Select 5 random users for recommendations
# random_users = np.random.choice(user_genre_profile.index, 5, replace=False)
random_users=[111,211,311,411,511]
print("Selected Users for Recommendation:", random_users)

# Provide top 5 movie recommendations per user based on their cluster
for user in random_users:
    # Get the cluster for the current user
    user_cluster = user_genre_profile.loc[user, 'Cluster']
    # Identify all users within the same cluster
    cluster_users = user_genre_profile[user_genre_profile['Cluster'] == user_cluster].index
    # Get movies that the current user has already rated
    user_rated_movies = set(ratings[ratings['userId'] == user]['movieId'])
    
    # Compute average ratings for movies by users in the same cluster
    cluster_ratings = ratings[ratings['userId'].isin(cluster_users)]
    cluster_movie_scores = cluster_ratings.groupby('movieId')['rating'].mean()
    
    # Exclude movies the current user has already rated
    recommended_movies = cluster_movie_scores.drop(user_rated_movies, errors='ignore')
    top_recommendations = recommended_movies.nlargest(5).index
    
    # Print user info and recommendations without the DataFrame index
    print(f"User {user} (Cluster: {user_cluster}) Top Recommendations:")
    print(movies[movies['movieId'].isin(top_recommendations)][['movieId', 'title']].to_string(index=False))
    print("\n")

    def evaluate_recommendations(user_id, top_recommendations, cluster_ratings):
        """Compute Precision@K, Recall@K, and NDCG"""
        relevant_movies = set(cluster_ratings[cluster_ratings['rating'] >= 5]['movieId'])
        recommended_movies = set(top_recommendations)
        
        precision = len(recommended_movies & relevant_movies) / best_k
        recall = len(recommended_movies & relevant_movies) / len(relevant_movies) if relevant_movies else 0
        
        # Compute NDCG
        true_relevance = [5 if movie in relevant_movies else 0 for movie in top_recommendations]
        ndcg = ndcg_score([true_relevance], [list(range(best_k, 0, -1))])
        
        return precision, recall, ndcg

# Generate recommendations and evaluate
for user in random_users:
    user_cluster = user_genre_profile.loc[user, 'Cluster']
    cluster_users = user_genre_profile[user_genre_profile['Cluster'] == user_cluster].index
    user_rated_movies = set(ratings[ratings['userId'] == user]['movieId'])
    
    cluster_ratings = ratings[ratings['userId'].isin(cluster_users)]
    cluster_movie_scores = cluster_ratings.groupby('movieId')['rating'].mean()
    recommended_movies = cluster_movie_scores.drop(user_rated_movies, errors='ignore').nlargest(best_k).index
    
    precision, recall, ndcg = evaluate_recommendations(user, recommended_movies, cluster_ratings)
    
    print(f"User {user} (Cluster {user_cluster}) - Precision@{best_k}: {precision:.4f}, Recall@{best_k}: {recall:.4f}, NDCG: {ndcg:.4f}")