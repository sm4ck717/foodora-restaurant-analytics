import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.neighbors import NearestNeighbors
from lightgbm import LGBMRanker
import warnings

warnings.filterwarnings('ignore')

# 1. Load Data
print("Loading data...")
df = pd.read_csv('final data.csv')
original_df = df.copy()

# ==========================================
# TASK 1: PREDICT QUALITY (CatBoost)
# ==========================================
print("Training Task 1: Regression Model...")
df1 = original_df.copy()

# Preprocessing
df1["Cuisines"] = df1["Cuisines"].str.replace(", ", " | ").astype(str)
categorical_cols = ["Country", "City", "Locality", "Cuisines", "Price range",
                    "Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"]

for col in categorical_cols:
    df1[col] = df1[col].astype(str)

df1["Aggregate rating"] = pd.to_numeric(df1["Aggregate rating"], errors="coerce")
df1 = df1.dropna(subset=["Aggregate rating"])

features_t1 = ["Country", "City", "Locality", "Cuisines", "Price range",
               "Has Table booking", "Has Online delivery", "Is delivering now",
               "Switch to order menu", "Votes", "Longitude", "Latitude"]

X = df1[features_t1]
y = df1["Aggregate rating"]

cat_features_indices = [X.columns.get_loc(col) for col in categorical_cols]

# Train Model
model_t1 = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=10, loss_function="RMSE", verbose=0)
model_t1.fit(X, y, cat_features=cat_features_indices)
model_t1.save_model("catboost_model.cbm")

# Save Options for App (Cascading Dropdowns)
geo_map = df1.groupby("Country").apply(
    lambda x: x.groupby("City")["Locality"].unique().to_dict()
).to_dict()

options = {
    "Price range": sorted(df1["Price range"].unique().tolist()),
    "geo_map": geo_map,
    "features": features_t1
}

with open("task1_options.pkl", "wb") as f:
    pickle.dump(options, f)

# ==========================================
# TASK 2: CUISINE STRATEGY (Clustering)
# ==========================================
print("Training Task 2: Clustering...")
df2 = original_df.copy()
df2['Cuisines'] = df2['Cuisines'].astype(str)
df2_exp = df2.assign(Cuisines=df2['Cuisines'].str.split(r'\s*\|\s*|,\s*', regex=True)).explode("Cuisines")
df2_exp['Cuisines'] = df2_exp['Cuisines'].str.strip()

for col in ["Aggregate rating", "Votes", "Price range"]:
    df2_exp[col] = pd.to_numeric(df2_exp[col], errors='coerce')

cuisine_stats = df2_exp.groupby("Cuisines").agg(
    avg_rating=("Aggregate rating", "mean"),
    avg_votes=("Votes", "mean"),
    avg_price=("Price range", "mean"),
    count=("Restaurant ID", "count")
).reset_index()

X_cluster = cuisine_stats[["avg_rating", "avg_votes", "avg_price"]]
scaler_t2 = StandardScaler()
X_scaled = scaler_t2.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=4, random_state=42)
cuisine_stats["cluster"] = kmeans.fit_predict(X_scaled)
cuisine_stats.to_csv("cuisine_clusters.csv", index=False)

# ==========================================
# TASK 3: RECOMMENDER
# ==========================================
print("Training Task 3: Recommender...")
df3 = original_df.copy()
df3 = df3.sort_values("Votes", ascending=False).drop_duplicates(subset=["Restaurant Name"], keep="first").reset_index(drop=True)
df3["Cuisines"] = df3["Cuisines"].fillna("Unknown")

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df3["Cuisines"])

scaler_t3 = MinMaxScaler()
num_features = df3[["Price range", "Aggregate rating", "Longitude", "Latitude"]].fillna(0)
num_scaled = scaler_t3.fit_transform(num_features)

combined_features = hstack([tfidf_matrix, num_scaled]).tocsr()

nn = NearestNeighbors(n_neighbors=11, metric="cosine")
nn.fit(combined_features)

# Precompute recommendations for App efficiency
recommendations = {}
distances, indices = nn.kneighbors(combined_features)
for i, name in enumerate(df3["Restaurant Name"]):
    rec_indices = indices[i][1:] # Skip self
    rec_names = df3.iloc[rec_indices]["Restaurant Name"].tolist()
    recommendations[name] = rec_names

df3[["Restaurant Name", "Cuisines", "Aggregate rating", "Price range", "City"]].to_csv("restaurants_for_rec.csv", index=False)
with open("recommendations.pkl", "wb") as f:
    pickle.dump(recommendations, f)

# ==========================================
# TASK 4: PARTNER RANKING (LightGBM Ranker)
# ==========================================
print("Training Task 4: Ranking...")

df4 = original_df.copy()

# Remove duplicate restaurants (keep most popular)
df4 = df4.sort_values("Votes", ascending=False) \
         .drop_duplicates(subset="Restaurant ID", keep="first")

# -----------------------------
# Feature Engineering
# -----------------------------
df4["Aggregate rating"] = pd.to_numeric(df4["Aggregate rating"], errors="coerce").fillna(0)
df4["Votes"] = pd.to_numeric(df4["Votes"], errors="coerce").fillna(0)
df4["Price range"] = pd.to_numeric(df4["Price range"], errors="coerce").fillna(1)

# Composite quality signal (used ONLY for label creation)
df4["quality_score"] = (
    0.7 * df4["Aggregate rating"] +
    0.3 * np.log1p(df4["Votes"])
)

# Convert to graded relevance (required for LambdaRank)
df4["quality_grade"] = pd.qcut(
    df4["quality_score"],
    q=4,
    labels=[0, 1, 2, 3]
).astype(int)

# -----------------------------
# Encode categorical variables
# -----------------------------
cat_cols_t4 = [
    "City", "Locality", "Cuisines",
    "Has Table booking", "Has Online delivery",
    "Is delivering now", "Switch to order menu"
]

df4_encoded = df4.copy()
for col in cat_cols_t4:
    le = LabelEncoder()
    df4_encoded[col] = le.fit_transform(df4_encoded[col].astype(str))

# -----------------------------
# Feature Set for Ranking
# -----------------------------
features_t4 = [
    "City", "Locality", "Cuisines",
    "Price range", "Aggregate rating", "Votes",
    "Longitude", "Latitude",
    "Has Table booking", "Has Online delivery",
    "Is delivering now", "Switch to order menu"
]

# Sort by group key (mandatory for LightGBM ranking)
df4_sorted = df4_encoded.sort_values("Locality")

X_rank = df4_sorted[features_t4]
y_rank = df4_sorted["quality_grade"]

group_counts = df4_sorted.groupby("Locality", sort=False).size().values

# -----------------------------
# Train Ranker
# -----------------------------
ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

ranker.fit(X_rank, y_rank, group=group_counts)

# -----------------------------
# Generate Ranking Scores
# -----------------------------
df4_sorted["pred_score"] = ranker.predict(X_rank)
df4_sorted["rank_in_locality"] = (
    df4_sorted.groupby("Locality")["pred_score"]
    .rank(ascending=False, method="dense")
)

# Merge back readable fields
df4_final = df4.loc[df4_sorted.index].copy()
df4_final["pred_score"] = df4_sorted["pred_score"]
df4_final["rank_in_locality"] = df4_sorted["rank_in_locality"]

df4_final.to_csv("ranking_results.csv", index=False)

# -----------------------------
# SAVE MODEL + FEATURES (IMPORTANT)
# -----------------------------
with open("lightgbm_ranker.pkl", "wb") as f:
    pickle.dump(ranker, f)

with open("task4_features.pkl", "wb") as f:
    pickle.dump(features_t4, f)

# Optional: Save feature importance snapshot
fi_rank = pd.DataFrame({
    "feature": features_t4,
    "importance": ranker.feature_importances_
}).sort_values("importance", ascending=False)

fi_rank.to_csv("task4_feature_importance.csv", index=False)

print("Task 4 completed: Ranking model & artifacts saved.")
