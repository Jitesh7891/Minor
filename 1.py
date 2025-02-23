import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import random


# Load MovieLens dataset (use correct paths)
movies = pd.read_csv("./ml-latest-small/ml-latest-small/movies.csv")  # Movie ID, title, genres
ratings = pd.read_csv("./ml-latest-small/ml-latest-small/ratings.csv")  # User ratings

# Preserve original genres as a new column before one-hot encoding
movies["genre_string"] = movies["genres"]

# One-Hot Encode Genres
genre_data = movies["genres"].str.get_dummies(sep='|')
movies = pd.concat([movies, genre_data], axis=1)
movies.drop(columns=["genres"], inplace=True)

# Merge ratings with movies (include genres info)
# First, create a user-movie rating matrix using ratings pivot
df = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
# Prepare genre features (one-hot encoded columns) from movies
genre_features = movies.set_index("movieId").drop(columns=["title", "genre_string"])

# Merge genre features into the rating matrix:
df = df.T  # Transpose so movieId is index
df = df.join(genre_features, how="left").fillna(0)
df = df.T  # Transpose back to original form

# Normalize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Find optimal K using Elbow Method & Silhouette Score
inertia = []
silhouette_scores = []
K_range = range(4, 10)  # Testing K from 2 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))

# Plot Elbow Method & Silhouette Score
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

# Select best K based on Silhouette Score
best_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal K: {best_k}")

# Apply K-Means with best K
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_scaled)
df["Cluster"] = clusters

def recommend_movies(user_id, top_n=5):
    if user_id not in df.index:
        return "User not found."
    
    user_cluster = df.loc[user_id, "Cluster"]
    similar_users = df[df["Cluster"] == user_cluster].drop(columns=["Cluster"])
    avg_movie_ratings = similar_users.mean().sort_values(ascending=False)
    
    # Get the movies the user has already rated
    user_rated_movies = ratings[ratings["userId"] == user_id]["movieId"].tolist()
    
    movie_recommendations = (
        avg_movie_ratings.reset_index()
        .merge(movies[["movieId", "title", "genre_string"]], on="movieId", how="left")
    )
    
    # Filter out movies already rated by the user
    movie_recommendations = movie_recommendations[~movie_recommendations["movieId"].isin(user_rated_movies)]
    
    # Rename and select desired columns
    movie_recommendations.rename(columns={"title": "Movie Name", "genre_string": "Category"}, inplace=True)
    return movie_recommendations[["Movie Name", "Category"]].head(top_n)

# Example Usage
# Generate 3 random user IDs
# random_user_ids = random.sample(range(1, 611), 5)

# # Get recommendations for each user
# for user_id in random_user_ids:
#     print(f"\nUser {user_id} belongs to Cluster {clusters[user_id]}")
#     print(f"Recommendations for User {user_id}:")
#     print(recommend_movies(user_id=user_id, top_n=5).to_string(index=False))

# Count the number of users in each cluster
cluster_counts = df["Cluster"].value_counts().sort_index()

# Display the counts
print(cluster_counts)


