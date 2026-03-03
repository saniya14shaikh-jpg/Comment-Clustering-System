"""
Sentiment & Emotion Analysis Engine
Uses VADER + TextBlob for sentiment scoring
Emotion detection using keyword matching
"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess, detect_sarcasm, detect_toxic

# Initialize analyzers
_VADER = SentimentIntensityAnalyzer()

# Emotion keyword mapping
EMOTION_KEYWORDS = {
    "happy": [
        "love", "amazing", "wonderful", "fantastic", "great",
        "awesome", "beautiful", "perfect", "excellent", "best",
        "happy", "joy", "smile", "laugh", "fun", "blessed",
        "grateful", "thankful", "inspired", "brilliant"
    ],
    "angry": [
        "hate", "angry", "furious", "rage", "mad", "terrible",
        "horrible", "awful", "disgusting", "pathetic", "worst",
        "stupid", "idiot", "moron", "garbage", "trash", "useless"
    ],
    "sad": [
        "sad", "cry", "crying", "tears", "heartbreak", "miss",
        "lonely", "depressed", "hurt", "pain", "grief", "loss",
        "broken", "empty", "hopeless", "miserable", "suffer"
    ],
    "excited": [
        "omg", "wow", "incredible", "unbelievable", "insane",
        "obsessed", "screaming", "cannot", "wait", "excited",
        "thrilled", "pumped", "hyped", "epic", "fire", "lit"
    ],
    "fear": [
        "scared", "afraid", "terrified", "horrified", "nightmare",
        "fear", "panic", "worried", "anxious", "nervous", "dread"
    ],
    "disgust": [
        "disgusting", "gross", "nasty", "revolting", "sick",
        "yuck", "eww", "vile", "repulsive", "awful", "filthy"
    ],
    "surprised": [
        "shocked", "surprised", "unexpected", "unbelievable",
        "woah", "whoa", "wait", "really", "seriously", "what"
    ],
    "neutral": [
        "okay", "fine", "average", "normal", "regular",
        "whatever", "alright", "guess", "suppose", "think"
    ]
}

def get_sentiment_vader(text):
    scores = _VADER.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return {
        "sentiment":       sentiment,
        "score":           round(compound, 4),
        "positive_score":  round(scores["pos"], 4),
        "negative_score":  round(scores["neg"], 4),
        "neutral_score":   round(scores["neu"], 4),
    }

def get_sentiment_textblob(text):
    analysis  = TextBlob(text)
    polarity  = round(analysis.sentiment.polarity, 4)
    subjectivity = round(analysis.sentiment.subjectivity, 4)
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return {
        "sentiment":    sentiment,
        "score":        polarity,
        "subjectivity": subjectivity,
    }

def detect_emotion(text):
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[emotion] = score
    best_emotion = max(scores, key=scores.get)
    if scores[best_emotion] == 0:
        best_emotion = "neutral"
    return best_emotion, scores

def get_confidence(vader_score, textblob_score):
    avg = (abs(vader_score) + abs(textblob_score)) / 2
    return round(min(avg + 0.3, 1.0), 4)

def analyze_comment(text):
    if not isinstance(text, str) or not text.strip():
        return {
            "sentiment":        "neutral",
            "emotion":          "neutral",
            "score":            0.0,
            "confidence":       0.0,
            "is_toxic":         False,
            "is_sarcastic":     False,
            "vader_score":      0.0,
            "textblob_score":   0.0,
        }

    # Get scores
    vader      = get_sentiment_vader(text)
    textblob   = get_sentiment_textblob(text)
    emotion, _ = detect_emotion(text)
    is_toxic   = detect_toxic(text)
    is_sarc    = detect_sarcasm(text)

    # Combine scores
    combined_score = round(
        (vader["score"] * 0.6) + (textblob["score"] * 0.4), 4)

    # Flip sentiment if sarcastic
    if is_sarc and combined_score > 0:
        combined_score = -combined_score

    # Final sentiment
    if combined_score >= 0.05:
        sentiment = "positive"
    elif combined_score <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Override if toxic
    if is_toxic:
        sentiment = "negative"
        emotion   = "angry"

    confidence = get_confidence(
        vader["score"], textblob["score"])

    return {
        "sentiment":      sentiment,
        "emotion":        emotion,
        "score":          combined_score,
        "confidence":     confidence,
        "is_toxic":       is_toxic,
        "is_sarcastic":   is_sarc,
        "vader_score":    vader["score"],
        "textblob_score": textblob["score"],
    }

def analyze_batch(comments):
    results = []
    for c in comments:
        result = analyze_comment(c)
        result["text"] = c
        results.append(result)
    return results

def get_summary(results):
    total    = len(results)
    positive = sum(1 for r in results if r["sentiment"] == "positive")
    negative = sum(1 for r in results if r["sentiment"] == "negative")
    neutral  = sum(1 for r in results if r["sentiment"] == "neutral")
    toxic    = sum(1 for r in results if r["is_toxic"])
    avg_score = round(
        sum(r["score"] for r in results) / max(total, 1), 4)
    avg_conf  = round(
        sum(r["confidence"] for r in results) / max(total, 1), 4)

    emotions = {}
    for r in results:
        e = r["emotion"]
        emotions[e] = emotions.get(e, 0) + 1

    return {
        "total":          total,
        "positive":       positive,
        "negative":       negative,
        "neutral":        neutral,
        "toxic":          toxic,
        "non_toxic":      total - toxic,
        "avg_score":      avg_score,
        "avg_confidence": avg_conf,
        "emotions":       emotions,
        "overall_sentiment": (
            "positive" if positive > negative
            else "negative" if negative > positive
            else "neutral"
        )
    }

if __name__ == "__main__":
    test_comments = [
        "This is absolutely amazing! Love it so much! ❤️",
        "You are such an idiot! Nobody likes you!",
        "Okay I guess. Nothing special.",
        "OMG I cannot believe this is real!! 🔥🔥",
        "Yeah right! Sure sure! Obviously perfect! 🙄",
        "This made me cry so much 😢 So beautiful",
    ]
    print("=" * 60)
    for comment in test_comments:
        result = analyze_comment(comment)
        print(f"Comment:   {comment[:50]}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Emotion:   {result['emotion'].upper()}")
        print(f"Score:     {result['score']}")
        print(f"Toxic:     {result['is_toxic']}")
        print(f"Sarcastic: {result['is_sarcastic']}")
        print("-" * 60)
