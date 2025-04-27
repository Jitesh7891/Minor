import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# ——— Your model and helpers ———
class AttributeFusedSVD:
    def __init__(self, n_factors=20, lr=0.005, reg=0.02, n_epochs=20, alpha=10, beta=10):
        self.n_factors, self.lr, self.reg = n_factors, lr, reg
        self.n_epochs, self.alpha, self.beta = n_epochs, alpha, beta

    def fit(self, R, U_attr, I_attr):
        n_u, n_i = R.shape
        self.U = np.random.normal(scale=0.1, size=(n_u, self.n_factors))
        self.V = np.random.normal(scale=0.1, size=(n_i, self.n_factors))
        self.A = np.random.normal(scale=0.1, size=(U_attr.shape[1], self.n_factors))
        self.B = np.random.normal(scale=0.1, size=(I_attr.shape[1], self.n_factors))
        mask = R > 0
        for _ in range(self.n_epochs):
            for i, j in zip(*mask.nonzero()):
                e = R[i,j] - self.U[i].dot(self.V[j])
                self.U[i] += self.lr*(2*e*self.V[j] - 2*self.reg*self.U[i])
                self.V[j] += self.lr*(2*e*self.U[i] - 2*self.reg*self.V[j])
            u_err = self.U.dot(self.A.T) - U_attr
            v_err = self.V.dot(self.B.T) - I_attr
            self.A -= self.lr*(2*self.alpha*u_err.T.dot(self.U) + 2*self.reg*self.A)
            self.B -= self.lr*(2*self.beta * v_err.T.dot(self.V) + 2*self.reg*self.B)
        return self

    def full_prediction(self):
        return self.U.dot(self.V.T)

def calculate_rmse(preds, actual):
    mask = actual > 0
    return np.sqrt(mean_squared_error(actual[mask], preds[mask]))

def load_data(path='d:/Minor/ml-100k/ml-100k'):
    # Ratings
    df = pd.read_csv(f'{path}/u.data', sep='\t',
                     names=['user','item','rating','ts'], nrows=1000)
    # Users
    u_user = pd.read_csv(f'{path}/u.user', sep='|',
                         names=['user','age','gender','occ','zip'])
    # Items
    u_item = pd.read_csv(f'{path}/u.item', sep='|',
                         names=['item','title','release','video','imdb'] + list(range(19)),
                         encoding='latin-1')
    # One-hot attrs
    enc_u, enc_i = OneHotEncoder(), OneHotEncoder()
    U_attr = enc_u.fit_transform(u_user[['age','gender','occ']]).toarray()
    I_attr = enc_i.fit_transform(u_item[list(range(19))]).toarray()
    # Build R matrix
    n_u, n_i = u_user.user.max(), u_item.item.max()
    R = np.zeros((n_u, n_i))
    R[df['user']-1, df['item']-1] = df['rating']
    return R, U_attr, I_attr, u_user, u_item

# ——— Main ———
if __name__ == '__main__':
    # Load data
    R, U_attr, I_attr, u_user, u_item = load_data()

    # Train model (you can replace with tuned params)
    model = AttributeFusedSVD()
    model.fit(R, U_attr, I_attr)
    P = model.full_prediction()

    # Select one random user who has rated something
    valid = [u for u in range(R.shape[0]) if np.count_nonzero(R[u])>0]
    random.seed(42)
    chosen = random.choice(valid)
    print(f"Selected user ID: {chosen+1}\n")

    # Compute top-5 recommendations for that user
    unrated = np.where(R[chosen]==0)[0]
    scores  = P[chosen, unrated]
    top_idxs = unrated[np.argsort(scores)[::-1][:5]]

    # 1) User info + recommended titles
    user_info = u_user.rename(columns={'user':'user_id'})
    user_info['user_id'] = user_info['user_id'].astype(int)
    titles = [u_item.iloc[i]['title'] for i in top_idxs]
    df_user_recs = pd.DataFrame([{
        'user_id': chosen+1,
        'age':      int(user_info.loc[chosen, 'age']),
        'gender':   user_info.loc[chosen, 'gender'],
        'occ':      user_info.loc[chosen, 'occ'],
        'recommended_movies': "; ".join(titles)
    }])
    print("=== User Info & Recommendations ===")
    print(df_user_recs.to_string(index=False), "\n")

    # 2) Details for each recommended movie
    rec_items = [i+1 for i in top_idxs]
    df_movies = u_item[u_item['item'].isin(rec_items)][['item','title','release']]
    print("=== Recommended Movies Info ===")
    print(df_movies.to_string(index=False))
