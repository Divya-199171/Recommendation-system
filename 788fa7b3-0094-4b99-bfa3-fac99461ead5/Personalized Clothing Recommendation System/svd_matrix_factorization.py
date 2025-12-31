import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

print("=" * 70)
print("MATRIX FACTORIZATION USING SVD")
print("=" * 70)

# Use train/test data from previous block
# Use normalized ratings (mean-centered)
_train_mean = train_matrix_cf[train_matrix_cf > 0].stack().mean()
print(f"\nGlobal mean rating: {_train_mean:.4f}")

# Create mean-centered matrix
train_matrix_centered = train_matrix_cf.copy()
train_matrix_centered[train_matrix_centered > 0] -= _train_mean

# Perform SVD with k=20 latent factors
k_factors = 20
print(f"\n📊 Performing SVD with k={k_factors} latent factors...")

# Convert to numpy for SVD
_train_matrix_np = train_matrix_centered.values

# Perform SVD
U, sigma, Vt = svds(_train_matrix_np, k=k_factors)

# Convert sigma to diagonal matrix
sigma_diag = np.diag(sigma)

# Reconstruct the matrix: R ≈ U * Σ * V^T
predicted_ratings_centered = np.dot(np.dot(U, sigma_diag), Vt)
predicted_ratings_svd = predicted_ratings_centered + _train_mean

print(f"U shape (users x factors): {U.shape}")
print(f"Σ shape (factors x factors): {sigma_diag.shape}")
print(f"V^T shape (factors x items): {Vt.shape}")
print(f"Reconstructed matrix shape: {predicted_ratings_svd.shape}")

# Make predictions for test set
print("\n🎯 Generating SVD predictions for test set...")
svd_predictions = []
svd_actuals = []

for _, row in test_data_cf.iterrows():
    _u_idx = int(row['user_idx'])
    _i_idx = int(row['item_idx'])
    _actual = row['rating']
    
    # Get prediction from reconstructed matrix
    _pred = predicted_ratings_svd[_u_idx, _i_idx]
    
    # Clip predictions to valid rating range (assuming 0-5 scale)
    _pred = np.clip(_pred, 0, 5)
    
    svd_predictions.append(_pred)
    svd_actuals.append(_actual)

svd_predictions = np.array(svd_predictions)
svd_actuals = np.array(svd_actuals)

# Calculate metrics
rmse_svd = np.sqrt(np.mean((svd_predictions - svd_actuals) ** 2))
mae_svd = np.mean(np.abs(svd_predictions - svd_actuals))

print(f"\nSVD Matrix Factorization RMSE: {rmse_svd:.4f}")
print(f"SVD Matrix Factorization MAE: {mae_svd:.4f}")

# ========== RECOMMENDATION FUNCTION ==========
def get_topn_recommendations_svd(user_idx, n=10):
    """Get top-N recommendations for a user using SVD"""
    # Get all predictions for this user
    _user_ratings = predicted_ratings_svd[user_idx, :]
    
    # Get items user hasn't interacted with
    _user_rated_items = train_matrix_cf.iloc[user_idx, :].values > 0
    
    # Set already-rated items to -inf so they won't be recommended
    _user_ratings_copy = _user_ratings.copy()
    _user_ratings_copy[_user_rated_items] = -np.inf
    
    # Get top-N item indices
    _top_indices = np.argsort(_user_ratings_copy)[-n:][::-1]
    
    # Get item IDs and predicted ratings
    _item_ids = train_matrix_cf.columns[_top_indices].tolist()
    _predicted_ratings = _user_ratings[_top_indices]
    
    return list(zip(_item_ids, _predicted_ratings))

# Example recommendations
print("\n" + "=" * 70)
print("EXAMPLE TOP-10 RECOMMENDATIONS (SVD)")
print("=" * 70)

_sample_users = [0, 100, 200]
for _u_idx in _sample_users:
    _recs = get_topn_recommendations_svd(_u_idx, n=10)
    print(f"\nUser {_u_idx} (actual index in matrix):")
    for rank, (item_id, pred_rating) in enumerate(_recs, 1):
        print(f"  {rank}. Item {item_id}: predicted rating = {pred_rating:.3f}")

# ========== FINAL SUMMARY ==========
print("\n" + "=" * 70)
print("ALL METHODS COMPARISON")
print("=" * 70)
print(f"User-Based CF    : RMSE={rmse_userbased:.4f}, MAE={mae_userbased:.4f}")
print(f"Item-Based CF    : RMSE={rmse_itembased:.4f}, MAE={mae_itembased:.4f}")
print(f"SVD (k={k_factors})        : RMSE={rmse_svd:.4f}, MAE={mae_svd:.4f}")
print("=" * 70)
