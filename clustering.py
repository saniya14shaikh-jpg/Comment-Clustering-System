"""
K-Means Clustering Engine
Clusters comments into groups based on TF-IDF vectors.
Cluster emotion labels are derived from actual VADER sentiment scores —
NOT hardcoded — so labels always reflect the true sentiment of each cluster.
"""

import os
import sys
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_batch

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

_EMOTION_MAP = {}
analyzer = SentimentIntensityAnalyzer()


def _score_text(text: str) -> float:
    return analyzer.polarity_scores(text)["compound"]


def _label_from_score(avg_score: float, avg_toxic: float) -> str:
    if avg_toxic > 0.35 or avg_score <= -0.40:
        return "Toxic ☣️"
    if avg_score >= 0.35:
        return "Positive 😀"
    if avg_score >= 0.10:
        return "Inspirational 🤩"
    if avg_score <= -0.15:
        return "Negative 😡"
    return "Sad & Emotional 😢"


TOXIC_WORDS = {
    "hate", "idiot", "stupid", "dumb", "ugly", "loser", "kill",
    "die", "trash", "garbage", "pathetic", "disgusting", "worthless",
    "moron", "jerk", "freak", "bastard", "scum", "retard",
    "nobody", "worst", "horrible", "terrible", "awful", "useless",
    "shut", "cursed", "filth", "vile", "pig", "cow", "fat", "weirdo"
}


def _toxic_rate(texts):
    rates = []
    for t in texts:
        words = set(t.lower().split())
        text_lower = t.lower()
        has_toxic = bool(words & TOXIC_WORDS) or any(
            phrase in text_lower for phrase in ["shut up", "go away", "nobody likes"]
        )
        rates.append(1.0 if has_toxic else 0.0)
    return float(np.mean(rates)) if rates else 0.0


def get_vectorizer(max_features=5000):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )


def fit_clusters(texts, n_clusters=6):
    global _EMOTION_MAP
    print(f"🔧 Fitting K-Means with {n_clusters} clusters...")

    cleaned = preprocess_batch(texts)
    vectorizer = get_vectorizer()
    X = vectorizer.fit_transform(cleaned)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X)

    if len(texts) > n_clusters:
        try:
            sil_score = silhouette_score(X, kmeans.labels_)
            print(f"   Silhouette Score: {sil_score:.4f}")
        except Exception:
            pass

    # Build emotion map from REAL sentiment scores per cluster
    cluster_scores = {}
    cluster_toxic  = {}
    for cluster_id in range(n_clusters):
        indices = [i for i, lbl in enumerate(kmeans.labels_) if lbl == cluster_id]
        cluster_texts = [texts[i] for i in indices]
        if cluster_texts:
            scores = [_score_text(t) for t in cluster_texts]
            cluster_scores[cluster_id] = float(np.mean(scores))
            cluster_toxic[cluster_id]  = _toxic_rate(cluster_texts)
        else:
            cluster_scores[cluster_id] = 0.0
            cluster_toxic[cluster_id]  = 0.0

    _EMOTION_MAP = {
        cid: _label_from_score(cluster_scores[cid], cluster_toxic[cid])
        for cid in range(n_clusters)
    }

    print("   📌 Cluster → Emotion mapping:")
    for cid, emotion in _EMOTION_MAP.items():
        print(f"      Cluster {cid}: avg_score={cluster_scores[cid]:.3f} → {emotion}")

    joblib.dump(vectorizer,   f"{MODEL_DIR}/tfidf_vectorizer.pkl")
    joblib.dump(kmeans,       f"{MODEL_DIR}/kmeans_model.pkl")
    joblib.dump(_EMOTION_MAP, f"{MODEL_DIR}/emotion_map.pkl")
    print(f"   ✅ Models saved to {MODEL_DIR}/")

    return vectorizer, kmeans, X


def load_models():
    global _EMOTION_MAP
    vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.pkl")
    kmeans     = joblib.load(f"{MODEL_DIR}/kmeans_model.pkl")
    emotion_map_path = f"{MODEL_DIR}/emotion_map.pkl"
    if os.path.exists(emotion_map_path):
        _EMOTION_MAP = joblib.load(emotion_map_path)
    else:
        _EMOTION_MAP = {i: "Neutral 😐" for i in range(kmeans.n_clusters)}
    return vectorizer, kmeans


def generate_cluster_labels(vectorizer, kmeans, n_terms=5):
    terms  = vectorizer.get_feature_names_out()
    labels = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        top_idx   = center.argsort()[-n_terms:][::-1]
        top_words = [terms[j] for j in top_idx]
        labels[i] = f"Cluster {i}: {', '.join(top_words)}"
    return labels


def predict_cluster(text, vectorizer, kmeans, cluster_labels=None):
    cleaned  = preprocess_batch([text])
    X        = vectorizer.transform(cleaned)
    cluster  = int(kmeans.predict(X)[0])

    label = (cluster_labels.get(cluster, f"Cluster {cluster}")
             if cluster_labels else f"Cluster {cluster}")

    distances  = kmeans.transform(X)[0]
    confidence = round(1 / (1 + distances[cluster]), 4)

    emotion = _EMOTION_MAP.get(cluster, "Unknown 🤔")

    # Per-comment safety override
    own_score    = _score_text(text)
    own_toxic    = _toxic_rate([text])
    safe_emotion = _label_from_score(own_score, own_toxic)

    positive_labels = {"Positive 😀", "Inspirational 🤩"}

    # Comment is clearly positive but cluster says otherwise
    if own_score >= 0.35 and emotion not in positive_labels:
        emotion = safe_emotion

    # Comment is clearly negative/toxic but cluster says positive
    elif own_score <= -0.15 and emotion in positive_labels:
        emotion = safe_emotion

    # Comment has toxic words + negative score → always override
    elif own_toxic > 0.0 and own_score <= -0.10:
        emotion = safe_emotion

    return {
        "cluster":    cluster,
        "label":      label,
        "emotion":    emotion,
        "confidence": confidence,
        "own_score":  round(own_score, 4)
    }


def get_cluster_stats(labels, cluster_labels=None):
    stats = {}
    total = len(labels)
    for label in set(labels):
        count = list(labels).count(label)
        cluster_name = (cluster_labels.get(label, f"Cluster {label}")
                        if cluster_labels else f"Cluster {label}")
        stats[cluster_name] = {
            "count":      count,
            "percentage": round(count / total * 100, 2)
        }
    return stats


def get_pca_coords(X, labels, cluster_labels=None):
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())
    return [
        {
            "x":       float(coords[i][0]),
            "y":       float(coords[i][1]),
            "cluster": int(labels[i]),
            "label":   (cluster_labels.get(int(labels[i]), f"Cluster {labels[i]}")
                        if cluster_labels else f"Cluster {labels[i]}")
        }
        for i in range(len(labels))
    ]


def get_top_terms(vectorizer, kmeans, n_terms=10):
    terms    = vectorizer.get_feature_names_out()
    clusters = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        top_idx   = center.argsort()[-n_terms:][::-1]
        top_words = [terms[j] for j in top_idx]
        label     = f"Cluster {i}: {', '.join(top_words[:5])}"
        clusters[label] = top_words
    return clusters


if __name__ == "__main__":
    import pandas as pd
    print("📂 Loading dataset...")
    df    = pd.read_csv("instagram_comments.csv")
    texts = df["comment_text"].tolist()
    print(f"   {len(texts)} comments loaded")

    vectorizer, kmeans, X = fit_clusters(texts, n_clusters=6)
    cluster_labels = generate_cluster_labels(vectorizer, kmeans)

    test_cases = [
        "worst ever, absolutely terrible",
        "This is amazing, I love it so much!",
        "you are an idiot and I hate you",
        "so beautiful and inspiring!",
        "okay I guess, nothing special",
    ]
    print("\n🔍 Prediction Tests:")
    for t in test_cases:
        r = predict_cluster(t, vectorizer, kmeans, cluster_labels)
        print(f"   '{t[:50]}' → {r['emotion']}  (score={r['own_score']})")
