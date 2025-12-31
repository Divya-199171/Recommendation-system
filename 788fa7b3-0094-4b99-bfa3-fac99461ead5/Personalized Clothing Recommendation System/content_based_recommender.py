import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== BUILD PRODUCT CONTENT PROFILES =====
print("=" * 60)
print("CONTENT-BASED RECOMMENDER SYSTEM")
print("=" * 60)

# Create combined text features for each product (category, brand, color, style)
products_clean['content_features'] = (
    products_clean['category'] + ' ' + 
    products_clean['brand'] + ' ' + 
    products_clean['color'] + ' ' + 
    products_clean['style']
)

print("\n📝 PRODUCT CONTENT FEATURES:")
print(f"   Sample features: {products_clean['content_features'].head(3).tolist()}")

# Build TF-IDF matrix for product features
tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=None,  # No stop words needed for product attributes
    token_pattern=r'\w+'
)

tfidf_matrix = tfidf_vectorizer.fit_transform(products_clean['content_features'])

print(f"\n✅ TF-IDF Matrix built: {tfidf_matrix.shape} (products × features)")
print(f"   Feature vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
print(f"   Features: {list(tfidf_vectorizer.vocabulary_.keys())[:10]}...")

# Calculate product-product similarity matrix (cosine similarity)
product_similarity = cosine_similarity(tfidf_matrix)

print(f"\n✅ Product similarity matrix: {product_similarity.shape}")
print(f"   Average similarity: {product_similarity.mean():.4f}")
_max_sim = product_similarity.copy()
np.fill_diagonal(_max_sim, 0)
print(f"   Max similarity (excluding diagonal): {_max_sim.max():.4f}")

# ===== BUILD USER PREFERENCE PROFILES =====
print("\n" + "=" * 60)
print("USER PREFERENCE PROFILES FROM PURCHASE HISTORY")
print("=" * 60)

# Get purchase history for each user
user_profiles = []

for user_id in users_clean['user_id'].unique():
    # Get products this user purchased
    user_purchases = purchase_interactions[
        purchase_interactions['user_id'] == user_id
    ]['product_id'].unique()
    
    if len(user_purchases) > 0:
        # Get product indices
        product_indices = products_clean[
            products_clean['product_id'].isin(user_purchases)
        ].index.tolist()
        
        # Aggregate TF-IDF vectors of purchased products (mean) - convert to array
        user_tfidf = np.asarray(tfidf_matrix[product_indices].mean(axis=0)).flatten()
        
        # Get user's preferred style from profile
        user_style = users_clean[users_clean['user_id'] == user_id]['preferred_style'].values[0]
        
        user_profiles.append({
            'user_id': user_id,
            'tfidf_profile': user_tfidf,
            'purchased_products': user_purchases,
            'num_purchases': len(user_purchases),
            'preferred_style': user_style
        })

user_profiles_df = pd.DataFrame(user_profiles)

print(f"\n✅ User profiles created: {len(user_profiles_df)} users with purchase history")
print(f"   Average purchases per user: {user_profiles_df['num_purchases'].mean():.1f}")
print(f"   Users with 1 purchase: {(user_profiles_df['num_purchases'] == 1).sum()}")
print(f"   Users with 2+ purchases: {(user_profiles_df['num_purchases'] >= 2).sum()}")
print(f"   Users with 3+ purchases: {(user_profiles_df['num_purchases'] >= 3).sum()}")

print("\n📊 Style distribution in user profiles:")
for style, count in user_profiles_df['preferred_style'].value_counts().items():
    print(f"   {style}: {count} users")

# ===== GENERATE RECOMMENDATIONS =====
print("\n" + "=" * 60)
print("GENERATING CONTENT-BASED RECOMMENDATIONS")
print("=" * 60)

def get_content_recommendations(user_id, top_n=10):
    """Generate content-based recommendations for a user"""
    
    # Get user profile
    user_profile = user_profiles_df[user_profiles_df['user_id'] == user_id]
    
    if len(user_profile) == 0:
        return None  # User has no purchase history
    
    user_tfidf = user_profile.iloc[0]['tfidf_profile'].reshape(1, -1)
    purchased_products = user_profile.iloc[0]['purchased_products']
    
    # Calculate similarity between user profile and all products
    user_product_similarity = cosine_similarity(
        user_tfidf, 
        tfidf_matrix
    ).flatten()
    
    # Create recommendations dataframe
    recommendations_content = products_clean.copy()
    recommendations_content['similarity_score'] = user_product_similarity
    
    # Exclude already purchased products
    recommendations_content = recommendations_content[
        ~recommendations_content['product_id'].isin(purchased_products)
    ]
    
    # Sort by similarity score
    recommendations_content = recommendations_content.sort_values(
        'similarity_score', 
        ascending=False
    ).head(top_n)
    
    return recommendations_content[[
        'product_id', 'category', 'brand', 'color', 'style', 
        'price', 'similarity_score'
    ]]

# Generate recommendations for sample users
sample_users = user_profiles_df.head(5)['user_id'].tolist()

print(f"\n🎯 Sample recommendations for {len(sample_users)} users:\n")

sample_recommendations = {}

for user_id in sample_users:
    recs = get_content_recommendations(user_id, top_n=5)
    sample_recommendations[user_id] = recs
    
    # Get user info
    user_info = users_clean[users_clean['user_id'] == user_id].iloc[0]
    user_prefs = user_profiles_df[user_profiles_df['user_id'] == user_id].iloc[0]
    
    print(f"User: {user_id} | Style: {user_info['preferred_style']} | Purchased: {user_prefs['num_purchases']} items")
    print(f"{'Product':<10} {'Category':<10} {'Brand':<8} {'Style':<10} {'Similarity':>10}")
    print("-" * 60)
    
    for _, rec in recs.iterrows():
        print(f"{rec['product_id']:<10} {rec['category']:<10} {rec['brand']:<8} {rec['style']:<10} {rec['similarity_score']:>10.4f}")
    print()

print("=" * 60)
print("✅ CONTENT-BASED RECOMMENDER COMPLETE")
print("=" * 60)
print("\n📦 Generated assets:")
print(f"   • tfidf_vectorizer: Trained TF-IDF vectorizer")
print(f"   • tfidf_matrix: Product feature matrix ({tfidf_matrix.shape})")
print(f"   • product_similarity: Product-product similarity matrix")
print(f"   • user_profiles_df: User preference profiles ({len(user_profiles_df)} users)")
print(f"   • get_content_recommendations(): Function to generate recommendations")
print("\n🎯 System recommends similar items based on:")
print("   ✓ Product attributes (category, brand, color, style)")
print("   ✓ User purchase history aggregated into preference profile")
print("   ✓ Cosine similarity matching user preferences to products")