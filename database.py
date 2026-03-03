"""
Database Layer — SQLite via SQLAlchemy
Stores comments, predictions, clusters, analytics
"""

import os
from datetime import datetime
from sqlalchemy import (create_engine, Column, Integer, String,
                        Float, DateTime, Text, Boolean)
from sqlalchemy.orm import declarative_base, sessionmaker

DB_URL  = os.getenv("DATABASE_URL", "sqlite:///comments.db")
engine  = create_engine(DB_URL,
                        connect_args={"check_same_thread": False}
                        if "sqlite" in DB_URL else {})
Base    = declarative_base()
Session = sessionmaker(bind=engine)

class Comment(Base):
    __tablename__ = "comments"
    id            = Column(Integer, primary_key=True)
    comment_id    = Column(String(20))
    username      = Column(String(50))
    post_id       = Column(String(20))
    comment_text  = Column(Text)
    cleaned_text  = Column(Text)
    sentiment     = Column(String(10))
    emotion       = Column(String(20))
    score         = Column(Float)
    confidence    = Column(Float)
    is_toxic      = Column(Boolean, default=False)
    is_sarcastic  = Column(Boolean, default=False)
    cluster       = Column(Integer)
    cluster_label = Column(String(50))
    session_id    = Column(String(50))
    created_at    = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

def save_comment(data: dict):
    session = Session()
    try:
        rec = Comment(**{
            k: v for k, v in data.items()
            if hasattr(Comment, k)
        })
        session.add(rec)
        session.commit()
        return rec.id
    finally:
        session.close()

def save_batch(comments: list):
    session = Session()
    try:
        for data in comments:
            rec = Comment(**{
                k: v for k, v in data.items()
                if hasattr(Comment, k)
            })
            session.add(rec)
        session.commit()
    finally:
        session.close()

def get_all_comments(limit=200):
    session = Session()
    try:
        rows = session.query(Comment).order_by(
            Comment.created_at.desc()).limit(limit).all()
        return [_to_dict(r) for r in rows]
    finally:
        session.close()

def get_stats():
    session = Session()
    try:
        total    = session.query(Comment).count()
        positive = session.query(Comment).filter_by(sentiment="positive").count()
        negative = session.query(Comment).filter_by(sentiment="negative").count()
        neutral  = session.query(Comment).filter_by(sentiment="neutral").count()
        toxic    = session.query(Comment).filter_by(is_toxic=True).count()
        all_r    = session.query(Comment).all()
        avg_score = round(
            sum(r.score or 0 for r in all_r) / max(total, 1), 4)
        avg_conf  = round(
            sum(r.confidence or 0 for r in all_r) / max(total, 1), 4)

        emotions = {}
        for r in all_r:
            e = r.emotion or "neutral"
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
        }
    finally:
        session.close()

def get_by_username(username: str):
    session = Session()
    try:
        rows = session.query(Comment).filter_by(
            username=username).all()
        return [_to_dict(r) for r in rows]
    finally:
        session.close()

def get_toxic_comments(limit=50):
    session = Session()
    try:
        rows = session.query(Comment).filter_by(
            is_toxic=True).limit(limit).all()
        return [_to_dict(r) for r in rows]
    finally:
        session.close()

def _to_dict(r):
    return {
        "id":            r.id,
        "comment_id":    r.comment_id,
        "username":      r.username,
        "post_id":       r.post_id,
        "comment_text":  r.comment_text,
        "sentiment":     r.sentiment,
        "emotion":       r.emotion,
        "score":         r.score,
        "confidence":    r.confidence,
        "is_toxic":      r.is_toxic,
        "is_sarcastic":  r.is_sarcastic,
        "cluster":       r.cluster,
        "cluster_label": r.cluster_label,
        "created_at":    str(r.created_at),
    }

if __name__ == "__main__":
    print("✅ Database initialized successfully!")
    print(f"   Tables created: comments")
    stats = get_stats()
    print(f"   Total comments: {stats['total']}")
