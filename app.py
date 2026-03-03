"""
Flask REST API Backend for Comment Clustering System
"""

import os
import sys
import csv
import io
import uuid

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess, get_text_features
from sentiment     import analyze_comment, analyze_batch, get_summary
from database      import (save_comment, save_batch, get_all_comments,
                           get_stats, get_toxic_comments, get_by_username)

app = Flask(__name__,
            static_folder="../frontend",
            template_folder="../frontend/templates")
CORS(app)

MODEL_DIR = "models"

# ── Load clustering models ─────────────────────────────────────────────────────
vectorizer = None
kmeans     = None

def load_clustering():
    global vectorizer, kmeans
    try:
        from clustering import load_models
        vectorizer, kmeans = load_models()
        print("✅ Clustering models loaded!")
    except Exception as e:
        print(f"⚠️ Clustering models not found: {e}")

load_clustering()

# ── Helper ─────────────────────────────────────────────────────────────────────
def get_cluster_info(text):
    if vectorizer and kmeans:
        try:
            from clustering import predict_cluster
            return predict_cluster(text, vectorizer, kmeans)
        except:
            pass
    return {"cluster": 0, "label": "Unknown", "confidence": 0.0}

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("../frontend/templates", "index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status":  "ok",
        "message": "Comment Clustering System API running",
        "models":  vectorizer is not None
    })

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data     = request.get_json()
    text     = data.get("text", "").strip()
    username = data.get("username", "anonymous")
    post_id  = data.get("post_id", "POST001")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Analyze
    result  = analyze_comment(text)
    cluster = get_cluster_info(text)
    features = get_text_features(text)

    # Save to database
    save_comment({
        "username":      username,
        "post_id":       post_id,
        "comment_text":  text,
        "cleaned_text":  features["cleaned"],
        "sentiment":     result["sentiment"],
        "emotion":       result["emotion"],
        "score":         result["score"],
        "confidence":    result["confidence"],
        "is_toxic":      result["is_toxic"],
        "is_sarcastic":  result["is_sarcastic"],
        "cluster":       cluster["cluster"],
        "cluster_label": cluster["label"],
        "session_id":    str(uuid.uuid4()),
    })

    return jsonify({
        "text":          text,
        "username":      username,
        "sentiment":     result["sentiment"],
        "emotion":       result["emotion"],
        "score":         result["score"],
        "confidence":    result["confidence"],
        "is_toxic":      result["is_toxic"],
        "is_sarcastic":  result["is_sarcastic"],
        "cluster":       cluster["cluster"],
        "cluster_label": cluster["label"],
        "cleaned_text":  features["cleaned"],
        "emojis":        [str(e) for e in features["emojis"]],
        "word_count":    features["word_count"],
    })

@app.route("/api/analyze/batch", methods=["POST"])
def batch_analyze():
    data     = request.get_json()
    comments = data.get("comments", [])

    if not comments:
        return jsonify({"error": "No comments provided"}), 400
    if len(comments) > 200:
        return jsonify({"error": "Max 200 comments per batch"}), 400

    session_id = str(uuid.uuid4())
    results    = []

    for item in comments:
        if isinstance(item, dict):
            text     = item.get("text", "").strip()
            username = item.get("username", "anonymous")
            post_id  = item.get("post_id", "POST001")
        else:
            text     = str(item).strip()
            username = "anonymous"
            post_id  = "POST001"

        if not text:
            continue

        result   = analyze_comment(text)
        cluster  = get_cluster_info(text)
        features = get_text_features(text)

        row = {
            "username":      username,
            "post_id":       post_id,
            "comment_text":  text,
            "cleaned_text":  features["cleaned"],
            "sentiment":     result["sentiment"],
            "emotion":       result["emotion"],
            "score":         result["score"],
            "confidence":    result["confidence"],
            "is_toxic":      result["is_toxic"],
            "is_sarcastic":  result["is_sarcastic"],
            "cluster":       cluster["cluster"],
            "cluster_label": cluster["label"],
            "session_id":    session_id,
        }
        results.append(row)

    # Save batch
    save_batch(results)
    summary = get_summary(results)

    return jsonify({
        "session_id": session_id,
        "total":      len(results),
        "summary":    summary,
        "results":    results,
    })

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f       = request.files["file"]
    content = f.read().decode("utf-8")
    reader  = csv.DictReader(io.StringIO(content))

    comments = []
    for row in reader:
        text = ""
        user = "anonymous"
        for col in ["comment_text","text","comment","Comment"]:
            if col in row and row[col].strip():
                text = row[col].strip()
                break
        for col in ["username","user","Username","User"]:
            if col in row and row[col].strip():
                user = row[col].strip()
                break
        if text:
            comments.append({"text": text, "username": user})

    if not comments:
        return jsonify({"error": "No comment_text column found"}), 400

    comments = comments[:200]

    # Reuse batch endpoint logic
    request._cached_json = (
        {"comments": comments}, request.get_json)
    return batch_analyze()

@app.route("/api/history")
def history():
    limit = int(request.args.get("limit", 100))
    return jsonify(get_all_comments(limit))

@app.route("/api/stats")
def stats():
    return jsonify(get_stats())

@app.route("/api/toxic")
def toxic():
    return jsonify(get_toxic_comments())

@app.route("/api/user/<username>")
def user_comments(username):
    return jsonify(get_by_username(username))

@app.route("/api/preprocess", methods=["POST"])
def preprocess_endpoint():
    data = request.get_json()
    text = data.get("text", "")
    return jsonify(get_text_features(text))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)