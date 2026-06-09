import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(
    page_title="Amazon Reviews Sentiment Analyzer",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"

)

st.markdown("""
    <style>
    /* Main body background styling */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    /* Header formatting */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #ff9900;
    }
    div[data-testid="stMetricLabel"] {
        color: #888888 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        color: #ff9900 !important;
        font-weight: 700;
    }
    /* Streamlit Button */
    .stButton>button {
        background: linear-gradient(135deg, #ff9900 0%, #ff5500 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(255, 153, 0, 0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #ffaa22 0%, #ff6611 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 153, 0, 0.4);
    }
    .stButton>button:active {
        transform: translateY(0px);
    }
    /* Text Area custom styling */
    .stTextArea textarea {
        background-color: #1a1c23 !important;
        color: #ffffff !important;
        border: 1px solid #3a3f50 !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: #ff9900 !important;
        box-shadow: 0 0 10px rgba(255, 153, 0, 0.2) !important;
    }
    /* Sentiment Box display */
    .sentiment-card {
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        border-left: 6px solid;
    }
    .sentiment-positive {
        background-color: rgba(40, 167, 69, 0.1);
        border-color: #28a745;
        color: #28a745;
    }
    .sentiment-neutral {
        background-color: rgba(255, 193, 7, 0.1);
        border-color: #ffc107;
        color: #ffc107;
    }
    .sentiment-negative {
        background-color: rgba(220, 53, 69, 0.1);
        border-color: #dc3545;
        color: #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource

def load_model_artifacts():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

@st.cache_data

def load_sample_data():
    try:
        df = pd.read_csv("sampled_reviews.csv")
        return df
    except FileNotFoundError:
        return None

st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", ["Dashboard & EDA", "Sentiment Predictor"])

model, vectorizer = load_model_artifacts()
df_sample = load_sample_data()
if model is not None and vectorizer is not None:
    st.sidebar.success("Model loaded successfully!")

if page == "Dashboard & EDA":
    st.title("Amazon Reviews Sentiment Dashboard")
    st.markdown("Explore review sentiment trends.")
    if df_sample is None:
        st.warning("Sample dataset `sampled_reviews.csv` not found. Please run the model training script (`train.py`) first to generate the sample dataset.")
    else:
        total_reviews = len(df_sample)
        avg_score = df_sample['Score'].mean()
        df_sample['HelpfulnessRatio'] = np.where(
            df_sample['HelpfulnessDenominator'] > 0,
            df_sample['HelpfulnessNumerator'] / df_sample['HelpfulnessDenominator'],
            0
        )
        avg_helpfulness = df_sample[df_sample['HelpfulnessDenominator'] > 0]['HelpfulnessRatio'].mean() * 100
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sample Reviews", f"{total_reviews:,}")
        with col2:
            st.metric("Average Rating", f"{avg_score:.2f} / 5.0")
        with col3:
            st.metric("Avg Review Helpfulness", f"{avg_helpfulness:.1f}%")
        st.markdown("<br>", unsafe_allow_html=True)
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.subheader("⭐ Distribution of Star Ratings")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            sns.countplot(x='Score', data=df_sample, color='#ff9900', ax=ax)
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
        with col_plot2:
            st.subheader("💬 Sentiment Distribution")
            def get_sentiment_name(score):
                if score <= 2: return "Negative"
                elif score == 3: return "Neutral"
                else: return "Positive"
            sentiments = df_sample['Score'].apply(get_sentiment_name)
            sentiment_counts = sentiments.value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0e1117')
            colors = ['#28a745', '#ff9900', '#dc3545']
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                   colors=colors, textprops={'color': 'white'}, startangle=140)
            st.pyplot(fig)
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("☁️ Review Word Clouds")
        wc_sentiment = st.radio("Select Sentiment for Word Cloud", ["Positive Reviews (4-5 Stars)", "Negative Reviews (1-2 Stars)"], horizontal=True)
        if "Positive" in wc_sentiment:
            wc_text = " ".join(df_sample[df_sample['Score'] >= 4]['Text'].dropna().apply(clean_text))
            color_map = 'Greens'
        else:
            wc_text = " ".join(df_sample[df_sample['Score'] <= 2]['Text'].dropna().apply(clean_text))
            color_map = 'Reds'
        if wc_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='#0e1117',
                                  colormap=color_map, max_words=100).generate(wc_text)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            fig_wc.patch.set_facecolor('#0e1117')
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        else:
            st.info("No reviews available to generate word cloud.")

elif page == "Sentiment Predictor":
    st.title("Interactive Sentiment Predictor")
    st.markdown("Analyze review sentiment using machine learning.")
    if model is None or vectorizer is None:
        st.warning("Machine learning model and vectorizer not found. Please run the model training script (`train.py`) first to generate `model.pkl` and `vectorizer.pkl`.")
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        review_input = st.text_area("Write your product review here:", height=150,
                                    placeholder="Type a review (e.g., 'This product is absolutely wonderful! It arrived on time and works perfectly.')")
        analyze_btn = st.button("Analyze Sentiment")
        if analyze_btn or review_input:
            if not review_input.strip():
                st.info("Please enter some text to analyze.")
            else:
                cleaned_input = clean_text(review_input)
                vec_input = vectorizer.transform([cleaned_input])
                pred_class = model.predict(vec_input)[0]
                pred_probs = model.predict_proba(vec_input)[0]
                classes = ['Negative', 'Neutral', 'Positive']
                predicted_sentiment = classes[pred_class]
                confidence = pred_probs[pred_class] * 100
                if predicted_sentiment == 'Positive':
                    class_style = "sentiment-positive"
                    emoji = "🟢 Positive"
                elif predicted_sentiment == 'Neutral':
                    class_style = "sentiment-neutral"
                    emoji = "🟡 Neutral"
                else:
                    class_style = "sentiment-negative"
                    emoji = "🔴 Negative"
                st.markdown(f"""
                    <div class="sentiment-card {class_style}">
                        <h3>Sentiment Analysis Result: {emoji}</h3>
                        <p style="font-size: 18px; margin: 5px 0;">The review sentiment is classified as <strong>{predicted_sentiment}</strong> with a confidence score of <strong>{confidence:.1f}%</strong>.</p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Confidence Scores Breakdown")
                prob_df = pd.DataFrame({
                    'Sentiment': classes,
                    'Probability (%)': [p * 100 for p in pred_probs]
                })
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                colors_bar = ['#dc3545', '#ffc107', '#28a745']
                sns.barplot(x='Probability (%)', y='Sentiment', data=prob_df, palette=colors_bar, ax=ax)
                ax.set_xlim(0, 100)
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)

