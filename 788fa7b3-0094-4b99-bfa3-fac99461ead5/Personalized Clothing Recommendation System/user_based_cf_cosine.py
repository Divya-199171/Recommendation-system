import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Prepare train-test split for evaluation
print("=" * 70)
print("USER-BASED & ITEM-BASED COLLABORATIVE FILTERING")
print("=" * 70)

# Use the rating matrix for CF (ratings are explicit feedback)
rating_data = rating_matrix.copy()

# Create train and test sets (80-20 split)
# Get non-zero entries
user_indices, item_indices = rating_data.values.nonzero()
interactions_df_split = pd.DataFrame({
    'user_idx': user_indices,
    'item_idx': item_indices,
    'rating': [rating_data.values[i, j] for i, j in zip(user_indices, item_indices)]
})

train_data_cf, test_data_cf = train_test_split(interactions_df_split, test_size=0.2, random_state=42)

# Create train matrix
train_matrix_cf = pd.DataFrame(0.0, index=rating_data.index, columns=rating_data.columns)
for _, row in train_data_cf.iterrows():
    train_matrix_cf.iloc[int(row['user_idx']), int(row['item_idx'])] = row['rating']

print(f"\nTrain set size: {len(train_data_cf)} ratings")
print(f"Test set size: {len(test_data_cf)} ratings")
print(f"Users in matrix: {train_matrix_cf.shape[0]}")
print(f"Items in matrix: {train_matrix_cf.shape[1]}")
print(f"Sparsity: {(1 - train_matrix_cf.astype(bool).sum().sum() / (train_matrix_cf.shape[0] * train_matrix_cf.shape[1])) * 100:.2f}%")

# ========== USER-BASED CF ==========
print("\n" + "=" * 70)
print("1. USER-BASED COLLABORATIVE FILTERING (COSINE SIMILARITY)")
print("=" * 70)

# Compute user-user similarity matrix
print("\n📊 Computing user-user similarity matrix...")
user_user_similarity = cosine_similarity(train_matrix_cf.values)
user_sim_df = pd.DataFrame(
    user_user_similarity,
    index=train_matrix_cf.index,
    columns=train_matrix_cf.index
)
print(f"User similarity matrix shape: {user_sim_df.shape}")
print(f"Average similarity: {user_user_similarity[np.triu_indices_from(user_user_similarity, k=1)].mean():.4f}")

# Make predictions using user-based CF
print("\n🎯 Generating user-based CF predictions...")
userbased_predictions = []
userbased_actuals = []
_global_mean = train_matrix_cf[train_matrix_cf > 0].mean().mean()

for _, row in test_data_cf.iterrows():
    _u_idx = int(row['user_idx'])
    _i_idx = int(row['item_idx'])
    _actual = row['rating']
    
    # Get item ratings from all users
    _item_ratings = train_matrix_cf.iloc[:, _i_idx]
    _rated_users = _item_ratings[_item_ratings > 0].index.tolist()
    
    if len(_rated_users) == 0:
        _pred = _global_mean
    else:
        # Get similarities for target user
        _user_sims = user_sim_df.iloc[_u_idx, _rated_users].values
        _ratings = _item_ratings.loc[_rated_users].values
        
        # Get top-20 similar users
        if len(_rated_users) > 20:
            _top_indices = np.argsort(_user_sims)[-20:]
            _user_sims = _user_sims[_top_indices]
            _ratings = _ratings[_top_indices]
        
        if _user_sims.sum() == 0:
            _pred = _global_mean
        else:
            _pred = (_user_sims * _ratings).sum() / _user_sims.sum()
    
    userbased_predictions.append(_pred)
    userbased_actuals.append(_actual)

userbased_predictions = np.array(userbased_predictions)
userbased_actuals = np.array(userbased_actuals)

rmse_userbased = np.sqrt(np.mean((userbased_predictions - userbased_actuals) ** 2))
mae_userbased = np.mean(np.abs(userbased_predictions - userbased_actuals))

print(f"\nUser-Based CF RMSE: {rmse_userbased:.4f}")
print(f"User-Based CF MAE: {mae_userbased:.4f}")

# ========== ITEM-BASED CF ==========
print("\n" + "=" * 70)
print("2. ITEM-BASED COLLABORATIVE FILTERING (COSINE SIMILARITY)")
print("=" * 70)

# Compute item-item similarity matrix
print("\n📊 Computing item-item similarity matrix...")
item_item_similarity = cosine_similarity(train_matrix_cf.T.values)
item_sim_df = pd.DataFrame(
    item_item_similarity,
    index=train_matrix_cf.columns,
    columns=train_matrix_cf.columns
)
print(f"Item similarity matrix shape: {item_sim_df.shape}")
print(f"Average similarity: {item_item_similarity[np.triu_indices_from(item_item_similarity, k=1)].mean():.4f}")

# Make predictions using item-based CF
print("\n🎯 Generating item-based CF predictions...")
itembased_predictions = []
itembased_actuals = []

for _, row in test_data_cf.iterrows():
    _u_idx = int(row['user_idx'])
    _i_idx = int(row['item_idx'])
    _actual = row['rating']
    
    # Get user ratings for all items
    _user_ratings = train_matrix_cf.iloc[_u_idx, :]
    _rated_items = _user_ratings[_user_ratings > 0].index.tolist()
    
    if len(_rated_items) == 0:
        _pred = _global_mean
    else:
        # Get target item column index
        _target_col = train_matrix_cf.columns[_i_idx]
        
        # Get similarities between target item and rated items
        _item_sims = item_sim_df.loc[_target_col, _rated_items].values
        _ratings = _user_ratings.loc[_rated_items].values
        
        # Get top-20 similar items
        if len(_rated_items) > 20:
            _top_indices = np.argsort(_item_sims)[-20:]
            _item_sims = _item_sims[_top_indices]
            _ratings = _ratings[_top_indices]
        
        if _item_sims.sum() == 0:
            _pred = _global_mean
        else:
            _pred = (_item_sims * _ratings).sum() / _item_sims.sum()
    
    itembased_predictions.append(_pred)
    itembased_actuals.append(_actual)

itembased_predictions = np.array(itembased_predictions)
itembased_actuals = np.array(itembased_actuals)

rmse_itembased = np.sqrt(np.mean((itembased_predictions - itembased_actuals) ** 2))
mae_itembased = np.mean(np.abs(itembased_predictions - itembased_actuals))

print(f"\nItem-Based CF RMSE: {rmse_itembased:.4f}")
print(f"Item-Based CF MAE: {mae_itembased:.4f}")

# ========== SUMMARY ==========
print("\n" + "=" * 70)
print("COLLABORATIVE FILTERING COMPARISON")
print("=" * 70)
print(f"User-Based CF: RMSE={rmse_userbased:.4f}, MAE={mae_userbased:.4f}")
print(f"Item-Based CF: RMSE={rmse_itembased:.4f}, MAE={mae_itembased:.4f}")
print("=" * 70)
