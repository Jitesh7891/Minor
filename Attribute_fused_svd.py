import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

class AttributeFusedSVD:
    def __init__(self, n_factors=20, lr=0.005, reg=0.02, n_epochs=20, alpha=10, beta=10):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.beta = beta

    def fit(self, ratings, user_attrs, item_attrs):
        n_users, n_items = ratings.shape
        self.U = np.random.normal(scale=0.1, size=(n_users, self.n_factors))
        self.V = np.random.normal(scale=0.1, size=(n_items, self.n_factors))
        self.A = np.random.normal(scale=0.1, size=(user_attrs.shape[1], self.n_factors))
        self.B = np.random.normal(scale=0.1, size=(item_attrs.shape[1], self.n_factors))
        mask = ratings > 0
        for _ in range(self.n_epochs):
            for i, j in zip(*mask.nonzero()):
                e = ratings[i, j] - self.U[i].dot(self.V[j])
                u_grad = -2 * e * self.V[j] + 2 * self.reg * self.U[i]
                v_grad = -2 * e * self.U[i] + 2 * self.reg * self.V[j]
                self.U[i] -= self.lr * u_grad
                self.V[j] -= self.lr * v_grad
            u_attr_err = self.U.dot(self.A.T) - user_attrs
            v_attr_err = self.V.dot(self.B.T) - item_attrs
            self.A -= self.lr * (2 * self.alpha * u_attr_err.T.dot(self.U) + 2 * self.reg * self.A)
            self.B -= self.lr * (2 * self.beta * v_attr_err.T.dot(self.V) + 2 * self.reg * self.B)
        return self

    def predict(self, user_idx, item_idx):
        return self.U[user_idx].dot(self.V[item_idx])

    def full_prediction(self):
        return self.U.dot(self.V.T)

def load_ml100k(path='ml-100k'):
    df = pd.read_csv(f'{path}/u.data', sep='\t', names=['user', 'item', 'rating', 'ts'], nrows=1000)
    u_user = pd.read_csv(f'{path}/u.user', sep='|', names=['user', 'age', 'gender', 'occ', 'zip'])
    u_item = pd.read_csv(f'{path}/u.item', sep='|', names=['item', 'title', 'release', 'video', 'imdb'] + list(range(19)), encoding='latin-1')
    enc_user = OneHotEncoder()
    enc_item = OneHotEncoder()
    user_attrs = enc_user.fit_transform(u_user[['age', 'gender', 'occ']]).toarray()
    item_attrs = enc_item.fit_transform(u_item[list(range(19))]).toarray()
    n_users = u_user.user.max()
    n_items = u_item.item.max()
    R = np.zeros((n_users, n_items))
    R[df['user'].values - 1, df['item'].values - 1] = df['rating'].values
    return R, user_attrs, item_attrs

def calculate_rmse(predictions, actual_ratings):
    mask = actual_ratings > 0
    predicted_ratings = predictions[mask]
    actual_ratings = actual_ratings[mask]
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    return rmse

def hyperparameter_tuning(R, user_attrs, item_attrs):
    param_grid = {
        'n_factors': [10, 20, 30],
        'lr': [0.001, 0.005, 0.01],
        'reg': [0.02, 0.05, 0.1],
        'n_epochs': [10, 20, 50],
    }
    
    best_rmse = float('inf')
    best_params = None
    
    for n_factors in param_grid['n_factors']:
        for lr in param_grid['lr']:
            for reg in param_grid['reg']:
                for n_epochs in param_grid['n_epochs']:
                    model = AttributeFusedSVD(n_factors=n_factors, lr=lr, reg=reg, n_epochs=n_epochs)
                    model.fit(R, user_attrs, item_attrs)
                    preds = model.full_prediction()
                    rmse = calculate_rmse(preds, R)
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = (n_factors, lr, reg, n_epochs)
    
    return best_params, best_rmse

def get_top_n_recommendations(predictions, R, n=5):
    top_n_recommendations = {}
    for user_idx in range(predictions.shape[0]):
        unrated_items = np.where(R[user_idx] == 0)[0]
        predicted_ratings = predictions[user_idx, unrated_items]
        top_n_items = unrated_items[np.argsort(predicted_ratings)[::-1][:n]]
        top_n_recommendations[user_idx] = top_n_items
    return top_n_recommendations

def print_recommendations(top_n_recommendations, u_item, user_idx=0):
    print(f"Top 5 Recommendations for User {user_idx + 1}:")
    for item_idx in top_n_recommendations[user_idx]:
        movie_title = u_item.iloc[item_idx]['title']
        print(f"- {movie_title}")
    print()

if __name__ == '__main__':
    R, U_attr, V_attr = load_ml100k('d:/Minor/ml-100k/ml-100k')
    
    # Perform hyperparameter tuning
    best_params, best_rmse = hyperparameter_tuning(R, U_attr, V_attr)
    print("Best Hyperparameters:", best_params)
    print("Best RMSE:", best_rmse)

    # Train model
    model = AttributeFusedSVD(n_factors=best_params[0], lr=best_params[1], reg=best_params[2], n_epochs=best_params[3])
    model.fit(R, U_attr, V_attr)
    predictions = model.full_prediction()

    # Get top N recommendations for each user
    top_n_recommendations = get_top_n_recommendations(predictions, R, n=5)

    # Load item data (movie titles)
    u_item = pd.read_csv('d:/Minor/ml-100k/ml-100k/u.item', sep='|', names=['item', 'title', 'release', 'video', 'imdb'] + list(range(19)), encoding='latin-1')

    # Print top 5 recommendations for User 1
    print_recommendations(top_n_recommendations, u_item, user_idx=208)  # Example for user 1
