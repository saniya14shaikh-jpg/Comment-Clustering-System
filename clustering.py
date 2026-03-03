"""
K-Means Clustering Engine
Clusters comments into groups based on TF-IDF vectors
"""

import os
import sys
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_batch

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Cluster labels mapping
CLUSTER_LABELS = {
    0: "Positive & Appreciative",
    1: "Negative & Critical",
    2: "Toxic & Harmful",
    3: "Excited & Enthusiastic",
    4: "Sad & Emotional",
    5: "Neutral & Questions",
    6: "Mixed Sentiment",
    7: "Spam & Irrelevant",
}

def get_vectorizer(max_features=5000):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )

def fit_clusters(texts, n_clusters=6):
    print(f"🔧 Fitting K-Means with {n_clusters} clusters...")

    # Preprocess
    cleaned = preprocess_batch(texts)

    # Vectorize
    vectorizer = get_vectorizer()
    X = vectorizer.fit_transform(cleaned)

    # K-Means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    kmeans.fit(X)

    # Silhouette score
    if len(texts) > n_clusters:
        try:
            sil_score = silhouette_score(X, kmeans.labels_)
            print(f"   Silhouette Score: {sil_score:.4f}")
        except:
            sil_score = 0.0

    # Save models
    joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.pkl")
    joblib.dump(kmeans,     f"{MODEL_DIR}/kmeans_model.pkl")
    print(f"   ✅ Models saved to {MODEL_DIR}/")

    return vectorizer, kmeans, X

def load_models():
    vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.pkl")
    kmeans     = joblib.load(f"{MODEL_DIR}/kmeans_model.pkl")
    return vectorizer, kmeans

def predict_cluster(text, vectorizer, kmeans):
    cleaned = preprocess_batch([text])
    X = vectorizer.transform(cleaned)
    cluster = int(kmeans.predict(X)[0])
    label   = CLUSTER_LABELS.get(cluster, f"Cluster {cluster}")
    
    # Distance to centroid as confidence
    distances  = kmeans.transform(X)[0]
    min_dist   = distances[cluster]
    confidence = round(1 / (1 + min_dist), 4)

    return {
        "cluster":    cluster,
        "label":      label,
        "confidence": confidence
    }

def get_cluster_stats(labels):
    stats = {}
    total = len(labels)
    for label in set(labels):
        count = list(labels).count(label)
        stats[CLUSTER_LABELS.get(label, f"Cluster {label}")] = {
            "count":      count,
            "percentage": round(count / total * 100, 2)
        }
    return stats

def get_pca_coords(X, labels):
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())
    return [
        {
            "x":       float(coords[i][0]),
            "y":       float(coords[i][1]),
            "cluster": int(labels[i]),
            "label":   CLUSTER_LABELS.get(int(labels[i]),
                       f"Cluster {labels[i]}")
        }
        for i in range(len(labels))
    ]

def get_top_terms(vectorizer, kmeans, n_terms=10):
    terms    = vectorizer.get_feature_names_out()
    clusters = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        top_idx   = center.argsort()[-n_terms:][::-1]
        top_words = [terms[j] for j in top_idx]
        clusters[CLUSTER_LABELS.get(i, f"Cluster {i}")] = top_words
    return clusters

if __name__ == "__main__":
    import pandas as pd
    print("📂 Loading dataset...")
    df  = pd.read_csv("instagram_comments.csv")
    texts = df["comment_text"].tolist()

    print(f"   {len(texts)} comments loaded")
    vectorizer, kmeans, X = fit_clusters(texts, n_clusters=6)

    # Show cluster distribution
    labels = kmeans.labels_
    stats  = get_cluster_stats(labels)
    print("\n📊 Cluster Distribution:")
    for cluster_name, info in stats.items():
        print(f"   {cluster_name}: {info['count']} ({info['percentage']}%)")

    # Show top terms per cluster
    print("\n🔤 Top Terms Per Cluster:")
    top_terms = get_top_terms(vectorizer, kmeans)
    for cluster_name, terms in top_terms.items():
        print(f"   {cluster_name}: {', '.join(terms[:5])}")
