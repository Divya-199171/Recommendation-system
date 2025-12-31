import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# ===== CLEAN USER DATA =====
users_clean = users_df.copy()

# Check for missing values
print("=" * 60)
print("DATA CLEANING & PREPROCESSING")
print("=" * 60)
print("\n🧹 USERS DATA CLEANING")
print(f"Missing values: {users_clean.isnull().sum().sum()}")
print(f"Duplicate user_ids: {users_clean['user_id'].duplicated().sum()}")

# Ensure age is reasonable
users_clean = users_clean[users_clean['age'].between(18, 100)]

# Create user age groups for segmentation
users_clean['age_group'] = pd.cut(users_clean['age'], 
                                    bins=[0, 25, 35, 50, 100], 
                                    labels=['18-25', '26-35', '36-50', '50+'])

# Normalize price sensitivity to 0-1 scale
users_clean['price_sensitivity_normalized'] = users_clean['price_sensitivity']

# Calculate days since registration
users_clean['days_since_registration'] = (pd.Timestamp.now() - users_clean['registration_date']).dt.days

print(f"✅ Cleaned users: {len(users_clean)} records")
print(f"   - Age groups created: {users_clean['age_group'].value_counts().to_dict()}")
print(f"   - Average days since registration: {users_clean['days_since_registration'].mean():.0f}")

# ===== CLEAN PRODUCT DATA =====
products_clean = products_df.copy()

print("\n🧹 PRODUCTS DATA CLEANING")
print(f"Missing values: {products_clean.isnull().sum().sum()}")
print(f"Duplicate product_ids: {products_clean['product_id'].duplicated().sum()}")

# Ensure price is positive
products_clean = products_clean[products_clean['price'] > 0]

# Create price tiers
products_clean['price_tier'] = pd.cut(products_clean['price'], 
                                       bins=[0, 50, 100, 200, 1000], 
                                       labels=['Budget', 'Mid-range', 'Premium', 'Luxury'])

# Calculate effective price after discount
products_clean['effective_price'] = products_clean['price'] * (1 - products_clean['discount_percent'] / 100)

# Flag out-of-stock items
products_clean['in_stock'] = products_clean['stock_quantity'] > 0

# Calculate days since launch
products_clean['days_since_launch'] = (pd.Timestamp.now() - products_clean['launch_date']).dt.days

print(f"✅ Cleaned products: {len(products_clean)} records")
print(f"   - Price tiers: {products_clean['price_tier'].value_counts().to_dict()}")
print(f"   - In stock: {products_clean['in_stock'].sum()} ({products_clean['in_stock'].mean()*100:.1f}%)")
print(f"   - Average effective price: ${products_clean['effective_price'].mean():.2f}")

# ===== CLEAN INTERACTION DATA =====
interactions_clean = interactions_df.copy()

print("\n🧹 INTERACTIONS DATA CLEANING")
print(f"Missing values in core fields: {interactions_clean[['user_id', 'product_id', 'interaction_type']].isnull().sum().sum()}")
print(f"Duplicate interactions: {interactions_clean.duplicated().sum()}")

# Remove any interactions with invalid user_ids or product_ids
valid_user_ids = set(users_clean['user_id'])
valid_product_ids = set(products_clean['product_id'])

interactions_clean = interactions_clean[
    interactions_clean['user_id'].isin(valid_user_ids) & 
    interactions_clean['product_id'].isin(valid_product_ids)
]

# Sort by timestamp
interactions_clean = interactions_clean.sort_values('timestamp').reset_index(drop=True)

# Create time-based features
interactions_clean['hour_of_day'] = interactions_clean['timestamp'].dt.hour
interactions_clean['day_of_week'] = interactions_clean['timestamp'].dt.dayofweek
interactions_clean['days_ago'] = (pd.Timestamp.now() - interactions_clean['timestamp']).dt.days

# Create interaction weights (purchases > ratings > clicks > views)
interaction_weights = {'view': 1, 'click': 2, 'purchase': 5, 'rating': 3}
interactions_clean['interaction_weight'] = interactions_clean['interaction_type'].map(interaction_weights)

print(f"✅ Cleaned interactions: {len(interactions_clean)} records")
print(f"   - Valid user-product pairs: 100%")
print(f"   - Interaction distribution: {interactions_clean['interaction_type'].value_counts().to_dict()}")
print(f"   - Date range: {interactions_clean['days_ago'].max()} to {interactions_clean['days_ago'].min()} days ago")

# ===== CREATE INTERACTION MATRICES =====
print("\n📊 CREATING INTERACTION MATRICES")

# Create user-product interaction matrix (for collaborative filtering)
# Using interaction weights for implicit feedback
user_product_matrix = interactions_clean.pivot_table(
    index='user_id',
    columns='product_id',
    values='interaction_weight',
    aggfunc='sum',
    fill_value=0
)

print(f"✅ User-Product Matrix: {user_product_matrix.shape} (users × products)")
print(f"   - Sparsity: {(1 - user_product_matrix.astype(bool).sum().sum() / (user_product_matrix.shape[0] * user_product_matrix.shape[1])) * 100:.2f}%")
print(f"   - Non-zero interactions: {user_product_matrix.astype(bool).sum().sum()}")

# Create purchase-only matrix (for explicit purchase prediction)
purchase_interactions = interactions_clean[interactions_clean['interaction_type'] == 'purchase']
purchase_matrix = purchase_interactions.pivot_table(
    index='user_id',
    columns='product_id',
    values='quantity',
    aggfunc='sum',
    fill_value=0
)

print(f"\n✅ Purchase Matrix: {purchase_matrix.shape} (users × products)")
print(f"   - Users with purchases: {(purchase_matrix.sum(axis=1) > 0).sum()}")
print(f"   - Products purchased: {(purchase_matrix.sum(axis=0) > 0).sum()}")

# Create rating matrix (for rating prediction)
rating_interactions = interactions_clean[interactions_clean['interaction_type'] == 'rating'].dropna(subset=['rating'])
rating_matrix = rating_interactions.pivot_table(
    index='user_id',
    columns='product_id',
    values='rating',
    aggfunc='mean',
    fill_value=0
)

print(f"\n✅ Rating Matrix: {rating_matrix.shape} (users × products)")
print(f"   - Users who rated: {(rating_matrix > 0).any(axis=1).sum()}")
print(f"   - Products rated: {(rating_matrix > 0).any(axis=0).sum()}")
print(f"   - Average rating: {rating_interactions['rating'].mean():.2f}")

# Convert to sparse matrices for memory efficiency
user_product_sparse = csr_matrix(user_product_matrix.values)
purchase_sparse = csr_matrix(purchase_matrix.values)
rating_sparse = csr_matrix(rating_matrix.values)

print("\n" + "=" * 60)
print("✅ DATA PREPROCESSING COMPLETE")
print("=" * 60)
print("\n📦 READY FOR RECOMMENDATION ALGORITHMS:")
print(f"   • {len(users_clean)} user profiles with demographics & preferences")
print(f"   • {len(products_clean)} products with attributes & pricing")
print(f"   • {len(interactions_clean)} user interactions with temporal features")
print(f"   • 3 interaction matrices (weighted, purchase, rating)")
print(f"   • Sparse matrix representations for efficient computation")
print("\n🎯 Datasets are clean and ready for collaborative filtering,")
print("   content-based filtering, and hybrid recommendation systems!")