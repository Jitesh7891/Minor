import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Model and utility functions
class AttributeFusedSVD:
    def __init__(self, n_factors=20, lr=0.005, reg=0.02, n_epochs=20, alpha=10, beta=10):
        self.n_factors, self.lr, self.reg = n_factors, lr, reg
        self.n_epochs, self.alpha, self.beta = n_epochs, alpha, beta

    def fit(self, R, U_attr, I_attr):
        np.random.seed(42)
        n_u, n_i = R.shape
        self.U = np.random.normal(scale=0.1, size=(n_u, self.n_factors))
        self.V = np.random.normal(scale=0.1, size=(n_i, self.n_factors))
        self.A = np.random.normal(scale=0.1, size=(U_attr.shape[1], self.n_factors))
        self.B = np.random.normal(scale=0.1, size=(I_attr.shape[1], self.n_factors))
        mask = R > 0
        for _ in range(self.n_epochs):
            for i, j in zip(*mask.nonzero()):
                e = R[i, j] - self.U[i].dot(self.V[j])
                self.U[i] += self.lr * (2 * e * self.V[j] - 2 * self.reg * self.U[i])
                self.V[j] += self.lr * (2 * e * self.U[i] - 2 * self.reg * self.V[j])
            u_err = self.U.dot(self.A.T) - U_attr
            v_err = self.V.dot(self.B.T) - I_attr
            self.A -= self.lr * (2 * self.alpha * u_err.T.dot(self.U) + 2 * self.reg * self.A)
            self.B -= self.lr * (2 * self.beta * v_err.T.dot(self.V) + 2 * self.reg * self.B)
        return self

    def full_prediction(self):
        return self.U.dot(self.V.T)

def calculate_rmse(preds, actual):
    mask = actual > 0
    return np.sqrt(mean_squared_error(actual[mask], preds[mask]))

def load_ml100k(path='d:/Minor/ml-100k/ml-100k'):
    df = pd.read_csv(f'{path}/u.data', sep='\t',
                     names=['user','item','rating','ts'], nrows=1000)
    u_user = pd.read_csv(f'{path}/u.user', sep='|',
                         names=['user','age','gender','occ','zip'])
    u_item = pd.read_csv(f'{path}/u.item', sep='|',
                         names=['item','title','release','video','imdb'] + list(range(19)),
                         encoding='latin-1')
    enc_u, enc_i = OneHotEncoder(), OneHotEncoder()
    U_attr = enc_u.fit_transform(u_user[['age','gender','occ']]).toarray()
    I_attr = enc_i.fit_transform(u_item[list(range(19))]).toarray()
    n_u, n_i = u_user.user.max(), u_item.item.max()
    R = np.zeros((n_u, n_i))
    R[df['user']-1, df['item']-1] = df['rating']
    return R, U_attr, I_attr

# Load data
R, U_attr, I_attr = load_ml100k()

# Hyperparameter tuning
def hyperparameter_tuning(R, U_attr, I_attr):
    grid = {
        'n_factors': [10, 20, 30],
        'lr': [0.001, 0.005, 0.01],
        'reg': [0.02, 0.1],
        'n_epochs': [10, 20, 50],
    }
    best_rmse, best_params = float('inf'), None
    for nf in grid['n_factors']:
        for lr in grid['lr']:
            for reg in grid['reg']:
                for ne in grid['n_epochs']:
                    model = AttributeFusedSVD(nf, lr, reg, ne)
                    model.fit(R, U_attr, I_attr)
                    P = model.full_prediction()
                    cur_rmse = calculate_rmse(P, R)
                    if cur_rmse < best_rmse:
                        best_rmse, best_params = cur_rmse, (nf, lr, reg, ne)
    return best_params, best_rmse

best_params, global_rmse = hyperparameter_tuning(R, U_attr, I_attr)
print(f"Global best RMSE: {global_rmse:.6f} with params {best_params}")

# Train final model
model = AttributeFusedSVD(*best_params)
model.fit(R, U_attr, I_attr)
P = model.full_prediction()

# Compute RMSE for groups excluding zero-rating users
rating_counts = (R > 0).sum(axis=1)
nonzero_users = np.where(rating_counts > 0)[0]
counts_nonzero = rating_counts[nonzero_users]
threshold = np.median(counts_nonzero)

warm = nonzero_users[counts_nonzero > threshold]
cold = nonzero_users[counts_nonzero <= threshold]

rmse_overall = global_rmse
rmse_warm = calculate_rmse(P[warm], R[warm]) if len(warm) > 0 else np.nan
rmse_cold = calculate_rmse(P[cold], R[cold]) if len(cold) > 0 else np.nan

# Print group RMSEs
print(f"Overall RMSE: {rmse_overall:.6f}")
print(f"Warm Users RMSE: {rmse_warm:.6f}")
print(f"Cold Users RMSE: {rmse_cold:.6f}")

# Plot comparison line
groups = ['Overall', 'Warm', 'Cold']
rmses = [rmse_overall, rmse_warm, rmse_cold]

plt.figure(figsize=(8,5))
plt.plot(groups, rmses, marker='o', linestyle='-')
plt.xlabel('User Group')
plt.ylabel('RMSE')
plt.title('RMSE Comparison: Overall vs Warm vs Cold Users')
plt.tight_layout()
plt.show()
