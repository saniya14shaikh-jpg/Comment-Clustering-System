import subprocess
import sys
import os

subprocess.run([sys.executable, "-m", "pip", "install",
    "nltk", "scikit-learn", "torch", "joblib", "SQLAlchemy",
    "tqdm", "vaderSentiment", "textblob", "emoji",
    "plotly", "pandas", "numpy"], capture_output=True)

import nltk
nltk.download('punkt',      quiet=True)
nltk.download('stopwords',  quiet=True)
nltk.download('wordnet',    quiet=True)
nltk.download('omw-1.4',    quiet=True)
nltk.download('punkt_tab',  quiet=True)

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, BACKEND_DIR)
os.chdir(BASE_DIR)

st.set_page_config(
    page_title="Comment Clustering System",
    page_icon="💬",
    layout="wide"
)

@st.cache_resource
def load_everything():
    csv_path = os.path.join(BASE_DIR, "instagram_comments.csv")
    if not os.path.exists(csv_path):
        sys.path.insert(0, os.path.join(BASE_DIR, "data"))
        from generate_dataset import generate_dataset
        generate_dataset(200, csv_path)

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    if not os.path.exists(os.path.join(models_dir, "kmeans_model.pkl")):
        from clustering import fit_clusters
        df = pd.read_csv(csv_path)
        fit_clusters(df["comment_text"].tolist(), n_clusters=6)

    from sentiment  import analyze_comment, analyze_batch, get_summary
    from clustering import load_models, predict_cluster
    vectorizer, kmeans = load_models()
    return analyze_comment, analyze_batch, get_summary, vectorizer, kmeans, predict_cluster

with st.spinner("🔄 Loading AI Models..."):
    try:
        analyze_comment, analyze_batch, get_summary, \
        vectorizer, kmeans, predict_cluster = load_everything()
        st.success("✅ Models loaded!")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

st.title("💬 Comment Clustering System")
st.markdown("**Instagram Comment Clustering & Toxicity Analysis Platform**")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Single Comment",
    "📋 Bulk Analysis",
    "📤 CSV Upload",
    "📊 Analytics"
])

def sentiment_color(s):
    if s == "positive": return "🟢"
    if s == "negative": return "🔴"
    return "🟡"

def emotion_emoji(e):
    m = {"happy":"😊","angry":"😡","sad":"😢","excited":"🤩",
         "fear":"😨","disgust":"🤢","surprised":"😲","neutral":"😐"}
    return m.get(e,"😐")

def get_cluster(text, sentiment_result=None):
    try:
        result = predict_cluster(text, vectorizer, kmeans)

        if sentiment_result:
            is_toxic  = sentiment_result.get("is_toxic", False)
            sentiment = sentiment_result.get("sentiment", "")
            score     = float(sentiment_result.get("score", 0))
            emotion   = result["emotion"]

            positive_labels = {"Positive 😀", "Inspirational 🤩"}

            SAD_WORDS = {"cry", "crying", "miss", "lonely", "alone", "sad", "hurt",
                         "hurts", "memories", "wish", "lost", "heartbroken", "tears",
                         "painful", "grief", "mourn", "sorrow", "regret", "longing"}

            text_words     = set(text.lower().split())
            is_sad_comment = bool(text_words & SAD_WORDS)

            if is_toxic:
                result["emotion"] = "Toxic ☣️"
            elif is_sad_comment:
                result["emotion"] = "Sad & Emotional 😢"
            elif sentiment == "negative" and score <= -0.30 and emotion in positive_labels:
                result["emotion"] = "Negative 😡"
            elif sentiment == "positive" and score >= 0.35 and emotion not in positive_labels:
                result["emotion"] = "Positive 😀"

        return result
    except Exception as ex:
        return {"cluster": 0, "label": "Unknown", "confidence": 0.0, "emotion": "Unknown"}

# ── Tab 1 Single ───────────────────────────────────────────────
with tab1:
    st.subheader("Analyze a Single Comment")
    username = st.text_input("Username:", placeholder="e.g. sarah_j")
    text     = st.text_area(
        "Enter Instagram comment:",
        height=120,
        placeholder="This is absolutely amazing! Love it so much! ❤️"
    )
    if st.button("🧠 Analyze Comment", key="btn1"):
        if text.strip():
            with st.spinner("Analyzing..."):
                result  = analyze_comment(text)
                cluster = get_cluster(text, sentiment_result=result)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sentiment",
                f"{sentiment_color(result['sentiment'])} {result['sentiment'].upper()}")
            col2.metric("Emotion",
                f"{emotion_emoji(result['emotion'])} {result['emotion'].upper()}")
            col3.metric("Score (-1 to +1)", result['score'])
            col4.metric("Confidence", f"{result['confidence']*100:.1f}%")

            col5, col6, col7 = st.columns(3)
            col5.metric("Toxic",     "☣️ YES" if result['is_toxic']     else "✅ NO")
            col6.metric("Sarcastic", "🙄 YES" if result['is_sarcastic'] else "✅ NO")
            col7.metric("Cluster",   f"{cluster['emotion']} ({cluster['label']})")

            if result['is_toxic']:
                st.error(f"☣️ TOXIC COMMENT DETECTED from @{username or 'anonymous'}")
            elif result['sentiment'] == 'positive':
                st.success(f"✅ Positive comment from @{username or 'anonymous'}!")
            elif result['sentiment'] == 'negative':
                st.error(f"❌ Negative comment from @{username or 'anonymous'}")
            else:
                st.warning(f"➖ Neutral comment from @{username or 'anonymous'}")
        else:
            st.warning("Please enter a comment!")

# ── Tab 2 Bulk ─────────────────────────────────────────────────
with tab2:
    st.subheader("Bulk Comment Analysis")
    st.markdown("Format: `username: comment` — one per line")
    bulk_input = st.text_area(
        "Enter comments:",
        height=250,
        placeholder="sarah_j: This is absolutely amazing! Love it! ❤️\nmike_23: You are such an idiot! Nobody likes you!\npriya_k: Okay I guess. Nothing special.\nalex_99: OMG I cannot believe this is real!!\nemma_w: This made me cry so much 😢"
    )
    if st.button("🚀 Analyze All", key="btn2"):
        lines = [l.strip() for l in bulk_input.split('\n') if l.strip()]
        if lines:
            comments, usernames = [], []
            for line in lines:
                idx = line.find(':')
                if idx > 0 and idx < 30:
                    usernames.append(line[:idx].strip())
                    comments.append(line[idx+1:].strip())
                else:
                    usernames.append("anonymous")
                    comments.append(line)

            with st.spinner(f"Analyzing {len(comments)} comments..."):
                results = analyze_batch(comments)
                summary = get_summary(results)

            st.markdown("### 📊 Summary")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total",    summary['total'])
            c2.metric("Positive", summary['positive'])
            c3.metric("Negative", summary['negative'])
            c4.metric("Neutral",  summary['neutral'])
            c5.metric("Toxic ☣️", summary['toxic'])

            c6, c7, c8 = st.columns(3)
            c6.metric("Avg Score",      summary['avg_score'])
            c7.metric("Avg Confidence", f"{summary['avg_confidence']*100:.1f}%")
            c8.metric("Overall",        summary['overall_sentiment'].upper())

            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.pie(
                    values=[summary['positive'], summary['negative'], summary['neutral']],
                    names=['Positive', 'Negative', 'Neutral'],
                    color_discrete_sequence=['#22c55e', '#ef4444', '#f59e0b'],
                    title="Sentiment Distribution"
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                fig2 = px.pie(
                    values=[summary['non_toxic'], summary['toxic']],
                    names=['Safe', 'Toxic'],
                    color_discrete_sequence=['#22c55e', '#dc2626'],
                    title="Toxicity Distribution"
                )
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig2, use_container_width=True)

            if summary['emotions']:
                fig3 = px.bar(
                    x=list(summary['emotions'].keys()),
                    y=list(summary['emotions'].values()),
                    color_discrete_sequence=['#8b5cf6'],
                    title="Emotion Distribution",
                    labels={'x': 'Emotion', 'y': 'Count'}
                )
                fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig3, use_container_width=True)

            st.markdown("### 📋 Detailed Results")
            df_results = pd.DataFrame([{
                "Username":   usernames[i],
                "Comment":    comments[i][:80],
                "Sentiment":  f"{sentiment_color(r['sentiment'])} {r['sentiment'].upper()}",
                "Emotion":    f"{emotion_emoji(r['emotion'])} {r['emotion']}",
                "Score":      r['score'],
                "Confidence": f"{r['confidence']*100:.1f}%",
                "Toxic":      "☣️ YES" if r['is_toxic'] else "✅ NO",
                "Sarcastic":  "🙄 YES" if r['is_sarcastic'] else "NO",
            } for i, r in enumerate(results)])
            st.dataframe(df_results, use_container_width=True)

            toxic_list = [(usernames[i], comments[i])
                          for i, r in enumerate(results) if r['is_toxic']]
            if toxic_list:
                st.markdown("### ☣️ Toxic Comments Detected")
                for user, comment in toxic_list:
                    st.error(f"@{user}: {comment}")
        else:
            st.warning("Enter at least one comment!")

# ── Tab 3 CSV ──────────────────────────────────────────────────
with tab3:
    st.subheader("Upload CSV File")
    st.markdown("CSV must have **comment_text** column and optionally **username**")
    uploaded = st.file_uploader("Choose CSV:", type="csv")
    if uploaded:
        df_in = pd.read_csv(uploaded)
        st.write(f"Loaded: {len(df_in)} rows")
        st.dataframe(df_in.head(5), use_container_width=True)
        if st.button("📊 Analyze CSV", key="btn3"):
            col = next((c for c in
                ["comment_text", "text", "comment", "Comment"]
                if c in df_in.columns), None)
            ucol = next((c for c in
                ["username", "user", "Username"]
                if c in df_in.columns), None)
            if col:
                comments  = df_in[col].dropna().astype(str).tolist()[:200]
                usernames = (df_in[ucol].astype(str).tolist()[:200]
                             if ucol else ["anonymous"] * len(comments))
                with st.spinner(f"Processing {len(comments)} comments..."):
                    results = analyze_batch(comments)
                    summary = get_summary(results)
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total",    summary['total'])
                c2.metric("Positive", summary['positive'])
                c3.metric("Negative", summary['negative'])
                c4.metric("Neutral",  summary['neutral'])
                c5.metric("Toxic",    summary['toxic'])
                df_out = df_in.head(len(results)).copy()
                df_out['sentiment']  = [r['sentiment']  for r in results]
                df_out['emotion']    = [r['emotion']    for r in results]
                df_out['score']      = [r['score']      for r in results]
                df_out['confidence'] = [r['confidence'] for r in results]
                df_out['is_toxic']   = [r['is_toxic']   for r in results]
                st.dataframe(df_out, use_container_width=True)
                st.download_button(
                    "⬇️ Download Results",
                    df_out.to_csv(index=False),
                    "comment_results.csv"
                )
            else:
                st.error("No comment_text column found!")

# ── Tab 4 Analytics ────────────────────────────────────────────
with tab4:
    st.subheader("📊 Analytics Dashboard")
    if st.button("🔄 Load Analytics", key="btn4"):
        try:
            sys.path.insert(0, BACKEND_DIR)
            from database import get_stats, get_all_comments
            stats = get_stats()
            hist  = get_all_comments(200)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total",    stats['total'])
            c2.metric("Positive", stats['positive'])
            c3.metric("Negative", stats['negative'])
            c4.metric("Toxic",    stats['toxic'])

            if hist:
                df_h = pd.DataFrame(hist)

                col_a, col_b = st.columns(2)
                with col_a:
                    fig = px.pie(
                        values=[stats['positive'], stats['negative'], stats['neutral']],
                        names=['Positive', 'Negative', 'Neutral'],
                        color_discrete_sequence=['#22c55e', '#ef4444', '#f59e0b'],
                        title="Sentiment Distribution"
                    )
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig, use_container_width=True)

                with col_b:
                    if stats['emotions']:
                        fig2 = px.bar(
                            x=list(stats['emotions'].keys()),
                            y=list(stats['emotions'].values()),
                            color_discrete_sequence=['#8b5cf6'],
                            title="Emotion Distribution"
                        )
                        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                        st.plotly_chart(fig2, use_container_width=True)

                st.dataframe(
                    df_h[['username', 'comment_text', 'sentiment',
                           'emotion', 'score', 'is_toxic']].head(50),
                    use_container_width=True)
            else:
                st.info("No data yet — analyze some comments first!")
        except Exception as e:
            st.error(f"Error loading analytics: {e}")

st.markdown("---")
st.markdown("💬 Comment Clustering System | K-Means + VADER + TextBlob")
