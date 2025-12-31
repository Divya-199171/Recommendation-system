import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 70)
print("HYBRID RECOMMENDER SYSTEM")
print("=" * 70)

# ========== WEIGHTED HYBRID APPROACH ==========
print("\n📊 Building Weighted Hybrid Recommender...")
print("   Combining collaborative filtering (SVD) + content-based approaches")

# Define hybrid weights (tunable hyperparameters)
cf_weight = 0.6  # Collaborative filtering weight
content_weight = 0.4  # Content-based weight

print(f"\n⚖️  Hybrid Weights:")
print(f"   CF (SVD): {cf_weight:.1%}")
print(f"   Content-based: {content_weight:.1%}")

def get_hybrid_recommendations(user_id, n=10, w_cf=0.6, w_content=0.4, cold_start_threshold=2):
    """
    Generate hybrid recommendations combining CF and content-based approaches
    
    Args:
        user_id: User ID to generate recommendations for
        n: Number of recommendations to return
        w_cf: Weight for collaborative filtering
        w_content: Weight for content-based filtering
        cold_start_threshold: Min interactions for CF (else pure content-based)
    
    Returns:
        DataFrame with hybrid recommendations
    """
    
    # Check if user has purchase history for content-based
    user_profile_exists = user_id in user_profiles_df['user_id'].values
    
    # Check if user exists in CF matrix (has ratings)
    user_in_cf = user_id in rating_data.index
    
    # Get user's interaction count
    user_interaction_count = 0
    if user_in_cf:
        user_idx_in_matrix = rating_data.index.get_loc(user_id)
        user_interaction_count = (rating_data.iloc[user_idx_in_matrix] > 0).sum()
    
    # COLD START HANDLING - NEW USERS
    if not user_profile_exists and not user_in_cf:
        print(f"\n❄️  COLD START: User {user_id} is completely new (no history)")
        print(f"   Strategy: Recommend popular items + diverse categories")
        
        # Popularity-based recommendations
        item_popularity = purchase_interactions.groupby('product_id').size().sort_values(ascending=False)
        popular_items = item_popularity.head(n * 2).index.tolist()
        
        # Get diverse products from popular items
        popular_products = products_clean[products_clean['product_id'].isin(popular_items)].copy()
        
        # Diversify by category
        diverse_recs = []
        seen_categories = set()
        for _, product in popular_products.iterrows():
            if len(diverse_recs) >= n:
                break
            if product['category'] not in seen_categories or len(diverse_recs) < n // 2:
                diverse_recs.append(product)
                seen_categories.add(product['category'])
        
        hybrid_recs = pd.DataFrame(diverse_recs)[['product_id', 'category', 'brand', 'style', 'price']].head(n)
        hybrid_recs['hybrid_score'] = 1.0 - (hybrid_recs.index * 0.05)  # Decreasing score
        hybrid_recs['method'] = 'cold_start_popular'
        
        return hybrid_recs
    
    # COLD START - NEW ITEMS (handled in recommendation ranking with recency bonus)
    
    # Determine strategy based on data availability
    if user_interaction_count < cold_start_threshold:
        # Sparse CF data → favor content-based
        strategy = "content_heavy"
        w_cf_adjusted = 0.3
        w_content_adjusted = 0.7
        print(f"\n🔍 Strategy: {strategy} (user has {user_interaction_count} interactions)")
    else:
        # Sufficient data → use specified weights
        strategy = "balanced"
        w_cf_adjusted = w_cf
        w_content_adjusted = w_content
        print(f"\n🔍 Strategy: {strategy} (user has {user_interaction_count} interactions)")
    
    # Initialize scores dictionary
    item_scores = {}
    
    # ===== COLLABORATIVE FILTERING SCORES (SVD) =====
    if user_in_cf:
        user_idx_in_matrix = rating_data.index.get_loc(user_id)
        
        # Get SVD predictions for all items
        cf_scores_array = predicted_ratings_svd[user_idx_in_matrix, :]
        
        # Normalize CF scores to [0, 1]
        cf_min, cf_max = cf_scores_array.min(), cf_scores_array.max()
        if cf_max > cf_min:
            cf_scores_normalized = (cf_scores_array - cf_min) / (cf_max - cf_min)
        else:
            cf_scores_normalized = np.zeros_like(cf_scores_array)
        
        # Map to product IDs
        for idx, product_id in enumerate(rating_data.columns):
            item_scores[product_id] = {'cf_score': cf_scores_normalized[idx]}
    
    # ===== CONTENT-BASED SCORES =====
    if user_profile_exists:
        user_profile = user_profiles_df[user_profiles_df['user_id'] == user_id].iloc[0]
        user_tfidf = user_profile['tfidf_profile'].reshape(1, -1)
        purchased_products = user_profile['purchased_products']
        
        # Calculate similarity scores for all products
        content_scores_array = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
        
        # Map to product IDs
        for idx, product_id in enumerate(products_clean['product_id']):
            if product_id not in item_scores:
                item_scores[product_id] = {}
            item_scores[product_id]['content_score'] = content_scores_array[idx]
            item_scores[product_id]['already_purchased'] = product_id in purchased_products
    
    # ===== COMBINE SCORES =====
    hybrid_scores_list = []
    
    for product_id, scores in item_scores.items():
        cf_score = scores.get('cf_score', 0)
        content_score = scores.get('content_score', 0)
        already_purchased = scores.get('already_purchased', False)
        
        # Skip already purchased items
        if already_purchased:
            continue
        
        # Skip items user has already rated
        if user_in_cf and rating_data.loc[user_id, product_id] > 0:
            continue
        
        # Weighted hybrid score
        hybrid_score = (w_cf_adjusted * cf_score) + (w_content_adjusted * content_score)
        
        # Add novelty/diversity bonus based on product recency
        product_info = products_clean[products_clean['product_id'] == product_id].iloc[0]
        novelty_bonus = 1.0 / (1 + product_info['days_since_launch'] / 365)  # Decay over year
        hybrid_score += 0.05 * novelty_bonus  # Small bonus for new items
        
        hybrid_scores_list.append({
            'product_id': product_id,
            'cf_score': cf_score,
            'content_score': content_score,
            'hybrid_score': hybrid_score,
            'category': product_info['category'],
            'brand': product_info['brand'],
            'style': product_info['style'],
            'price': product_info['price']
        })
    
    # Convert to DataFrame and rank
    hybrid_recs = pd.DataFrame(hybrid_scores_list)
    hybrid_recs = hybrid_recs.sort_values('hybrid_score', ascending=False)
    
    # Diversify recommendations
    diversified_recs = []
    seen_categories = set()
    
    # First pass: ensure category diversity (max 40% from same category)
    max_per_category = max(2, n // 3)
    category_counts = {}
    
    for _, rec in hybrid_recs.iterrows():
        category = rec['category']
        if category not in category_counts:
            category_counts[category] = 0
        
        if category_counts[category] < max_per_category or len(diversified_recs) >= n * 0.7:
            diversified_recs.append(rec)
            category_counts[category] += 1
            
        if len(diversified_recs) >= n:
            break
    
    # If not enough, add remaining top-scored items
    if len(diversified_recs) < n:
        for _, rec in hybrid_recs.iterrows():
            if len(diversified_recs) >= n:
                break
            if not any(d['product_id'] == rec['product_id'] for d in diversified_recs):
                diversified_recs.append(rec)
    
    final_recs = pd.DataFrame(diversified_recs).head(n)
    final_recs['method'] = strategy
    
    return final_recs[['product_id', 'category', 'brand', 'style', 'price', 
                       'cf_score', 'content_score', 'hybrid_score', 'method']]

print("\n✅ Hybrid recommendation function created: get_hybrid_recommendations()")

# ========== TEST HYBRID RECOMMENDATIONS ==========
print("\n" + "=" * 70)
print("TESTING HYBRID RECOMMENDATIONS")
print("=" * 70)

# Test on different user types
test_user_types = [
    ('existing_rich', rating_data.index[0]),  # User with many ratings
    ('existing_sparse', rating_data.index[-1]),  # User with few ratings
]

# Find a completely new user (not in rating or purchase data)
all_user_ids = set(users_clean['user_id'].tolist())
rating_user_ids = set(rating_data.index.tolist())
purchase_user_ids = set(purchase_interactions['user_id'].unique().tolist())
cold_start_users = all_user_ids - rating_user_ids - purchase_user_ids

if len(cold_start_users) > 0:
    test_user_types.append(('cold_start', list(cold_start_users)[0]))

print(f"\n🧪 Testing {len(test_user_types)} user scenarios:\n")

test_recommendations = {}

for user_type, user_id in test_user_types:
    print(f"\n{'='*60}")
    print(f"User Type: {user_type.upper()}")
    print(f"User ID: {user_id}")
    print(f"{'='*60}")
    
    # Generate recommendations
    recs = get_hybrid_recommendations(user_id, n=10)
    test_recommendations[user_type] = recs
    
    # Display recommendations
    print(f"\n🎯 Top 10 Hybrid Recommendations:")
    print(f"{'Rank':<5} {'Product':<10} {'Category':<12} {'Style':<10} {'CF Score':<10} {'Content':<10} {'Hybrid':<10}")
    print("-" * 80)
    
    for idx, rec in recs.iterrows():
        rank = idx + 1 if isinstance(idx, int) else list(recs.index).index(idx) + 1
        print(f"{rank:<5} {rec['product_id']:<10} {rec['category']:<12} {rec['style']:<10} "
              f"{rec.get('cf_score', 0):.4f}    {rec.get('content_score', 0):.4f}    "
              f"{rec['hybrid_score']:.4f}")
    
    # Category diversity
    category_dist = recs['category'].value_counts()
    print(f"\n📊 Category Distribution:")
    for cat, count in category_dist.items():
        print(f"   {cat}: {count} items ({count/len(recs)*100:.0f}%)")

print("\n" + "=" * 70)
print("✅ HYBRID RECOMMENDER SYSTEM COMPLETE")
print("=" * 70)
