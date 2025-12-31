import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# ===== USER PROFILES =====
num_users = 1000
user_ids = [f"U{str(i).zfill(5)}" for i in range(1, num_users + 1)]

users_df = pd.DataFrame({
    'user_id': user_ids,
    'age': np.random.randint(18, 70, num_users),
    'gender': np.random.choice(['M', 'F', 'Other'], num_users, p=[0.48, 0.48, 0.04]),
    'location': np.random.choice(['Urban', 'Suburban', 'Rural'], num_users, p=[0.5, 0.35, 0.15]),
    'income_level': np.random.choice(['Low', 'Medium', 'High'], num_users, p=[0.25, 0.50, 0.25]),
    'registration_date': [datetime.now() - timedelta(days=np.random.randint(1, 730)) for _ in range(num_users)],
    'preferred_style': np.random.choice(['Casual', 'Formal', 'Sporty', 'Trendy', 'Classic'], num_users),
    'price_sensitivity': np.random.uniform(0, 1, num_users)  # 0 = very price sensitive, 1 = not sensitive
})

# ===== PRODUCT CATALOG =====
num_products = 500
product_ids = [f"P{str(i).zfill(5)}" for i in range(1, num_products + 1)]

categories = ['T-Shirts', 'Jeans', 'Dresses', 'Jackets', 'Shoes', 'Sweaters', 'Shorts', 'Skirts', 'Accessories']
brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'BrandF', 'BrandG', 'BrandH']
colors = ['Black', 'White', 'Blue', 'Red', 'Green', 'Gray', 'Brown', 'Navy', 'Beige']
styles = ['Casual', 'Formal', 'Sporty', 'Trendy', 'Classic']

products_df = pd.DataFrame({
    'product_id': product_ids,
    'category': np.random.choice(categories, num_products),
    'brand': np.random.choice(brands, num_products),
    'color': np.random.choice(colors, num_products),
    'price': np.round(np.random.uniform(15.99, 299.99, num_products), 2),
    'style': np.random.choice(styles, num_products),
    'stock_quantity': np.random.randint(0, 500, num_products),
    'launch_date': [datetime.now() - timedelta(days=np.random.randint(1, 1095)) for _ in range(num_products)],
    'discount_percent': np.random.choice([0, 5, 10, 15, 20, 25, 30], num_products, p=[0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05])
})

# ===== USER INTERACTIONS =====
num_interactions = 15000

# Generate realistic interactions (users more likely to interact with items matching their preferences)
interaction_data = []

for _ in range(num_interactions):
    user_idx = np.random.randint(0, num_users)
    user = users_df.iloc[user_idx]
    
    # Users more likely to interact with products matching their style preference
    style_match_mask = products_df['style'] == user['preferred_style']
    if style_match_mask.any() and np.random.random() < 0.6:
        product_idx = np.random.choice(products_df[style_match_mask].index)
    else:
        product_idx = np.random.randint(0, num_products)
    
    product = products_df.iloc[product_idx]
    
    # Interaction type probabilities: view > click > purchase > rating
    interaction_type = np.random.choice(['view', 'click', 'purchase', 'rating'], p=[0.50, 0.30, 0.15, 0.05])
    
    timestamp = datetime.now() - timedelta(days=np.random.randint(0, 180))
    
    interaction = {
        'interaction_id': f"I{str(len(interaction_data)).zfill(6)}",
        'user_id': user['user_id'],
        'product_id': product['product_id'],
        'interaction_type': interaction_type,
        'timestamp': timestamp,
    }
    
    # Add purchase-specific details
    if interaction_type == 'purchase':
        interaction['quantity'] = np.random.randint(1, 4)
        interaction['total_amount'] = np.round(product['price'] * interaction['quantity'] * (1 - product['discount_percent']/100), 2)
    else:
        interaction['quantity'] = None
        interaction['total_amount'] = None
    
    # Add rating-specific details
    if interaction_type == 'rating':
        # Ratings influenced by price sensitivity and actual price
        base_rating = np.random.normal(4.0, 1.0)
        if user['price_sensitivity'] > 0.7 and product['price'] > 150:
            base_rating -= 0.5
        interaction['rating'] = np.clip(np.round(base_rating, 1), 1.0, 5.0)
    else:
        interaction['rating'] = None
    
    interaction_data.append(interaction)

interactions_df = pd.DataFrame(interaction_data)

print("=" * 60)
print("SYNTHETIC E-COMMERCE DATA GENERATION COMPLETE")
print("=" * 60)
print(f"\n📊 Users Dataset: {users_df.shape[0]} users with {users_df.shape[1]} attributes")
print(f"   - Age range: {users_df['age'].min()}-{users_df['age'].max()}")
print(f"   - Gender distribution: {users_df['gender'].value_counts().to_dict()}")
print(f"   - Preferred styles: {users_df['preferred_style'].value_counts().to_dict()}")

print(f"\n🛍️  Products Dataset: {products_df.shape[0]} products with {products_df.shape[1]} attributes")
print(f"   - Categories: {products_df['category'].nunique()} ({', '.join(products_df['category'].unique())})")
print(f"   - Price range: ${products_df['price'].min():.2f} - ${products_df['price'].max():.2f}")
print(f"   - Brands: {products_df['brand'].nunique()}")

print(f"\n🔄 Interactions Dataset: {interactions_df.shape[0]} interactions")
print(f"   - Interaction types: {interactions_df['interaction_type'].value_counts().to_dict()}")
print(f"   - Unique users with interactions: {interactions_df['user_id'].nunique()}")
print(f"   - Unique products with interactions: {interactions_df['product_id'].nunique()}")
print(f"   - Date range: {interactions_df['timestamp'].min().date()} to {interactions_df['timestamp'].max().date()}")

print("\n✅ Raw datasets ready for cleaning and preprocessing")