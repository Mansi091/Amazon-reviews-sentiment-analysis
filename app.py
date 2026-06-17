import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import json
import os
import plotly.express as px
from wordcloud import WordCloud
import nltk

st.set_page_config(
    page_title="Amazon Reviews Sentiment Analyzer",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap');
    
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-title {
        background: linear-gradient(135deg, #ff9900 0%, #ff5500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 153, 0, 0.4);
        box-shadow: 0 12px 40px 0 rgba(255, 153, 0, 0.15);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        color: #ff9900 !important;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #090b0e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #ff9900 0%, #ff5500 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 20px rgba(255, 153, 0, 0.25);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #ffaa22 0%, #ff6611 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 153, 0, 0.45);
        border: none;
    }
    
    .stButton>button:active {
        transform: translateY(0px);
    }
    
    .stTextArea textarea {
        background-color: #12151c !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        padding: 15px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #ff9900 !important;
        box-shadow: 0 0 12px rgba(255, 153, 0, 0.25) !important;
    }
    
    .sentiment-card {
        padding: 24px;
        border-radius: 16px;
        margin-top: 20px;
        border-left: 6px solid;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .sentiment-positive {
        background: rgba(40, 167, 69, 0.08);
        border-color: #28a745;
        border-top: 1px solid rgba(40, 167, 69, 0.15);
        border-right: 1px solid rgba(40, 167, 69, 0.15);
        border-bottom: 1px solid rgba(40, 167, 69, 0.15);
        color: #34d399;
    }
    
    .sentiment-neutral {
        background: rgba(255, 193, 7, 0.08);
        border-color: #ffc107;
        border-top: 1px solid rgba(255, 193, 7, 0.15);
        border-right: 1px solid rgba(255, 193, 7, 0.15);
        border-bottom: 1px solid rgba(255, 193, 7, 0.15);
        color: #fbbf24;
    }
    
    .sentiment-negative {
        background: rgba(220, 53, 69, 0.08);
        border-color: #dc3545;
        border-top: 1px solid rgba(220, 53, 69, 0.15);
        border-right: 1px solid rgba(220, 53, 69, 0.15);
        border-bottom: 1px solid rgba(220, 53, 69, 0.15);
        color: #f87171;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.02);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        transition: all 0.2s ease;
        padding: 0 16px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background-color: rgba(255, 255, 255, 0.03);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 153, 0, 0.15) !important;
        color: #ff9900 !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_nltk():
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/stopwords.zip')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    return WordNetLemmatizer(), set(stopwords.words('english'))

lemmatizer, stop_words = load_nltk()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_model_artifacts():
    try:
        models = {}
        paths = {
            "Logistic Regression": "model_logistic.pkl",
            "Naive Bayes": "model_naive_bayes.pkl",
            "Random Forest": "model_random_forest.pkl"
        }
        for name, path in paths.items():
            if os.path.exists(path):
                with open(path, "rb") as f:
                    models[name] = pickle.load(f)
            elif name == "Logistic Regression" and os.path.exists("model.pkl"):
                with open("model.pkl", "rb") as f:
                    models[name] = pickle.load(f)
                    
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return models, vectorizer
    except Exception:
        return {}, None

@st.cache_data(show_spinner="Loading sample reviews...")
def load_sample_data():
    try:
        df = pd.read_csv("sampled_reviews.csv")
        return df
    except FileNotFoundError:
        return None

@st.cache_data(show_spinner="Generating Word Cloud...")
def generate_wordcloud(text, color_map):
    if not text.strip():
        return None
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='#0d0f14',
        colormap=color_map, 
        max_words=100
    ).generate(text)
    return wordcloud.to_image()

models, vectorizer = load_model_artifacts()
df_sample = load_sample_data()

st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="margin: 0; color: #ff9900 !important; font-size: 1.8rem;">Amazon Sentiment</h2>
        <p style="color: #64748b; font-size: 0.9rem; margin-top: 5px;">Powered by Multiple Models</p>
    </div>
    """,
    unsafe_allow_html=True
)

if models and vectorizer is not None:
    st.sidebar.success(f"✅ Loaded {len(models)} models successfully!")
else:
    st.sidebar.warning("⚠️ Models not trained. Run train.py first.")

uploaded_file = st.sidebar.file_uploader("Upload custom CSV dataset", type=['csv'], key="uploader")
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        required_cols = ['Score', 'Summary', 'Text']
        if all(col in df_uploaded.columns for col in required_cols):
            st.sidebar.success("✅ Custom dataset uploaded!")
            df_sample = df_uploaded
        else:
            st.sidebar.error("❌ CSV must contain columns: Score, Summary, Text")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading CSV: {e}")

st.markdown("<h1 class='main-title'>Amazon Reviews Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-top:-10px; margin-bottom: 25px;'>Analyze customer feedback and visualize sentiment patterns dynamically.</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Analysis Dashboard", "🔮 Sentiment Predictor", "📈 Model Insights"])

with tab1:
    if df_sample is None:
        st.warning("Sample dataset `sampled_reviews.csv` not found. Please run the model training script (`train.py`) first to generate the sample dataset.")
    else:
        total_reviews = len(df_sample)
        avg_score = df_sample['Score'].mean()
        
        if 'HelpfulnessDenominator' in df_sample.columns and 'HelpfulnessNumerator' in df_sample.columns:
            df_sample['HelpfulnessRatio'] = np.where(
                df_sample['HelpfulnessDenominator'] > 0,
                df_sample['HelpfulnessNumerator'] / df_sample['HelpfulnessDenominator'],
                0
            )
            helpful_subset = df_sample[df_sample['HelpfulnessDenominator'] > 0]
            avg_helpfulness = helpful_subset['HelpfulnessRatio'].mean() * 100 if len(helpful_subset) > 0 else 0.0
        else:
            avg_helpfulness = 0.0
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews Analyzed", f"{total_reviews:,}")
        with col2:
            st.metric("Average Star Rating", f"{avg_score:.2f} / 5.0")
        with col3:
            st.metric("Avg Review Helpfulness", f"{avg_helpfulness:.1f}%")
            
        st.markdown("<br>", unsafe_allow_html=True)
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.subheader("⭐ Distribution of Star Ratings")
            score_counts = df_sample['Score'].value_counts().reset_index()
            score_counts.columns = ['Rating', 'Count']
            score_counts = score_counts.sort_values(by='Rating')
            
            fig_bar = px.bar(
                score_counts, 
                x='Rating', 
                y='Count',
                labels={'Rating': 'Star Rating', 'Count': 'Number of Reviews'},
                template='plotly_dark',
                color='Count',
                color_continuous_scale=['#ffa726', '#f57c00']
            )
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                xaxis=dict(tickmode='linear', tick0=1, dtick=1)
            )
            st.plotly_chart(fig_bar, width="stretch")
            
        with col_plot2:
            st.subheader("💬 Sentiment Distribution")
            
            def get_sentiment_name(score):
                if score <= 2: return "Negative"
                elif score == 3: return "Neutral"
                else: return "Positive"
                
            sentiments = df_sample['Score'].apply(get_sentiment_name)
            sentiment_counts = sentiments.value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            fig_pie = px.pie(
                sentiment_counts, 
                names='Sentiment', 
                values='Count',
                hole=0.45,
                color='Sentiment',
                color_discrete_map={'Positive': '#22c55e', 'Neutral': '#eab308', 'Negative': '#ef4444'},
                template='plotly_dark'
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, width="stretch")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if 'Time' in df_sample.columns:
            st.subheader("📈 Average Rating Trend over Time")
            df_sample['Date'] = pd.to_datetime(df_sample['Time'], unit='s')
            df_sample['Year'] = df_sample['Date'].dt.year
            yearly_avg = df_sample.groupby('Year')['Score'].mean().reset_index()
            
            fig_line = px.line(
                yearly_avg, 
                x='Year', 
                y='Score',
                labels={'Score': 'Average Rating', 'Year': 'Year'},
                template='plotly_dark',
                markers=True
            )
            fig_line.update_traces(
                line=dict(color='#ff9900', width=3),
                marker=dict(size=8, color='#ff5500')
            )
            fig_line.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=30, b=10),
                height=320,
                yaxis=dict(range=[1, 5])
            )
            st.plotly_chart(fig_line, width="stretch")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("☁️ Review Word Clouds")
        wc_sentiment = st.radio(
            "Select Sentiment for Word Cloud", 
            ["Positive Reviews (4-5 Stars)", "Negative Reviews (1-2 Stars)"], 
            horizontal=True,
            key="wc_radio"
        )
        
        if "Positive" in wc_sentiment:
            positive_df = df_sample[df_sample['Score'] >= 4]
            if 'Cleaned_Text' in positive_df.columns:
                sampled_text = positive_df['Cleaned_Text'].dropna().sample(
                    n=min(2000, len(positive_df)), 
                    random_state=42
                ).astype(str)
                wc_text = " ".join(sampled_text)
            else:
                sampled_text = positive_df['Text'].dropna().sample(
                    n=min(1000, len(positive_df)), 
                    random_state=42
                )
                wc_text = " ".join(sampled_text.apply(clean_text))
            color_map = 'Greens'
        else:
            negative_df = df_sample[df_sample['Score'] <= 2]
            if 'Cleaned_Text' in negative_df.columns:
                sampled_text = negative_df['Cleaned_Text'].dropna().sample(
                    n=min(2000, len(negative_df)), 
                    random_state=42
                ).astype(str)
                wc_text = " ".join(sampled_text)
            else:
                sampled_text = negative_df['Text'].dropna().sample(
                    n=min(1000, len(negative_df)), 
                    random_state=42
                )
                wc_text = " ".join(sampled_text.apply(clean_text))
            color_map = 'Reds'
            
        if wc_text.strip():
            wc_image = generate_wordcloud(wc_text, color_map)
            if wc_image is not None:
                st.image(wc_image, width="stretch")
            else:
                st.info("No reviews available to generate word cloud.")
        else:
            st.info("No reviews available to generate word cloud.")

with tab2:
    st.subheader("🔮 Interactive Sentiment Predictor")
    st.markdown("Enter a review text below, select a machine learning model, and predict the sentiment.")
    
    if not models or vectorizer is None:
        st.warning("Machine learning models and vectorizer not found. Please run the model training script (`train.py`) first to generate model pickle files.")
    else:
        selected_model_name = st.selectbox(
            "Choose Classification Model:",
            list(models.keys()),
            index=0
        )
        active_model = models[selected_model_name]
        
        with st.form("prediction_form"):
            review_input = st.text_area(
                "Write your product review here:", 
                height=150,
                placeholder="Type a review (e.g., 'This product is absolutely wonderful! It arrived on time and works perfectly.')"
            )
            submit_btn = st.form_submit_button("Analyze Sentiment")
            
        if submit_btn or (review_input.strip() and not submit_btn):
            if not review_input.strip():
                st.info("Please enter some text to analyze.")
            else:
                cleaned_input = clean_text(review_input)
                vec_input = vectorizer.transform([cleaned_input])
                pred_class = active_model.predict(vec_input)[0]
                pred_probs = active_model.predict_proba(vec_input)[0]
                
                classes = ['Negative', 'Neutral', 'Positive']
                predicted_sentiment = classes[pred_class]
                confidence = pred_probs[pred_class] * 100
                
                if predicted_sentiment == 'Positive':
                    class_style = "sentiment-positive"
                    emoji = "🟢 Positive"
                    text_color = "#22c55e"
                elif predicted_sentiment == 'Neutral':
                    class_style = "sentiment-neutral"
                    emoji = "🟡 Neutral"
                    text_color = "#eab308"
                else:
                    class_style = "sentiment-negative"
                    emoji = "🔴 Negative"
                    text_color = "#ef4444"
                    
                st.markdown(f"""
                    <div class="sentiment-card {class_style}">
                        <h3 style="margin-top:0; color: {text_color} !important;">Sentiment Analysis Result: {emoji}</h3>
                        <p style="font-size: 18px; margin: 10px 0 0 0;">The review sentiment is classified as <strong>{predicted_sentiment}</strong> with a confidence score of <strong>{confidence:.1f}%</strong>.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                prob_df = pd.DataFrame({
                    'Sentiment': classes,
                    'Probability (%)': [p * 100 for p in pred_probs]
                })
                
                fig_pred = px.bar(
                    prob_df,
                    x='Probability (%)',
                    y='Sentiment',
                    orientation='h',
                    color='Sentiment',
                    color_discrete_map={'Positive': '#22c55e', 'Neutral': '#eab308', 'Negative': '#ef4444'},
                    template='plotly_dark',
                    text='Probability (%)'
                )
                fig_pred.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside',
                    cliponaxis=False
                )
                fig_pred.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    margin=dict(l=10, r=40, t=10, b=10),
                    height=200,
                    xaxis=dict(range=[0, 110])
                )
                st.plotly_chart(fig_pred, width="stretch")

with tab3:
    st.subheader("📈 Model Insights & Architecture")
    st.markdown("Here is the background on classifier configurations and a performance comparison between all trained models.")
    
    if os.path.exists("model_comparison.json"):
        try:
            with open("model_comparison.json", "r") as f:
                comparison_data = json.load(f)
                
            comp_list = []
            for m_name, m_metrics in comparison_data.items():
                comp_list.append({
                    "Model": m_name,
                    "Accuracy (%)": m_metrics["Accuracy"] * 100,
                    "Macro F1-Score (%)": m_metrics["Macro_F1"] * 100
                })
            comp_df = pd.DataFrame(comp_list)
            
            st.subheader("📊 Performance Comparison Chart")
            
            fig_comp = px.bar(
                comp_df,
                x='Model',
                y=['Accuracy (%)', 'Macro F1-Score (%)'],
                barmode='group',
                labels={'value': 'Percentage (%)', 'variable': 'Metric'},
                color_discrete_sequence=['#ff9900', '#22c55e'],
                template='plotly_dark'
            )
            fig_comp.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=30, b=10),
                height=350
            )
            st.plotly_chart(fig_comp, width="stretch")
            
            st.subheader("📋 Detailed Classification Report")
            selected_report_name = st.selectbox(
                "Select Model to View Detailed Class Metrics:",
                list(comparison_data.keys()),
                key="report_selector"
            )
            
            report_dict = comparison_data[selected_report_name]["report"]
            report_rows = []
            for key, val in report_dict.items():
                if key in ['accuracy', 'macro avg', 'weighted avg']:
                    if key == 'accuracy':
                        report_rows.append({
                            "Class": "Overall Accuracy",
                            "Precision": np.nan,
                            "Recall": np.nan,
                            "F1-Score": val
                        })
                    else:
                        report_rows.append({
                            "Class": key.title(),
                            "Precision": val["precision"],
                            "Recall": val["recall"],
                            "F1-Score": val["f1-score"]
                        })
                else:
                    report_rows.append({
                        "Class": f"{key} Reviews",
                        "Precision": val["precision"],
                        "Recall": val["recall"],
                        "F1-Score": val["f1-score"]
                    })
                    
            report_df = pd.DataFrame(report_rows)
            st.dataframe(
                report_df.style.background_gradient(
                    cmap='Oranges',
                    subset=['Precision', 'Recall', 'F1-Score']
                ).format(precision=4, na_rep="-"),
                width="stretch",
                hide_index=True
            )
        except Exception as e:
            st.error(f"Error rendering comparison: {e}")
    else:
        st.info("No comparative model metadata found. Re-run train.py to generate performance logs.")
        
        col_inf1, col_inf2 = st.columns(2)
        with col_inf1:
            st.markdown(
                """
                <div class="glass-card">
                    <h4>Pipeline Architecture</h4>
                    <ul style="padding-left: 20px;">
                        <li><strong>Feature Extraction:</strong> TF-IDF Vectorization (Unigrams + Bigrams)</li>
                        <li><strong>Vocabulary Size:</strong> Limit of 25,000 top features</li>
                        <li><strong>Classifier:</strong> Logistic Regression, Naive Bayes, Random Forest</li>
                        <li><strong>Text Prep:</strong> Lemmatization + HTML tag cleaning + non-alpha character removal</li>
                    </ul>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with col_inf2:
            st.markdown(
                """
                <div class="glass-card">
                    <h4>Performance Highlights</h4>
                    <ul style="padding-left: 20px;">
                        <li><strong>Dataset Trained On:</strong> Stratified sample of 50,000 reviews</li>
                        <li><strong>Balanced class weights</strong> significantly boost Neutral (3-Star) review F1-scores.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
