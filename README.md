# Amazon Reviews Sentiment Analysis & Dashboard

An interactive, dark-themed Streamlit web dashboard and Machine Learning pipeline for predicting and exploring sentiment in Amazon fine food reviews. The model uses **TF-IDF Vectorization** (unigrams & bigrams) paired with a **Multinomial Logistic Regression** classifier with balanced class weights to achieve an overall test accuracy of **82.61%** while significantly boosting performance on minority classes.

---

## 🚀 Key Features

* **Interactive Web Dashboard**: Built with Streamlit, presenting key metrics, ratings distributions, and dynamic word clouds.
* **Ratings Trend Over Time**: An interactive line chart tracking average ratings across years.
* **Real-time Predictor**: Input custom product reviews to receive immediate positive, neutral, or negative classifications alongside confidence level breakdown charts.
* **Optimized Pipeline**: Designed to handle large-scale datasets efficiently by using stratified data sampling and caching model/vectorizer artifacts.
* **NLP**: Performs HTML cleaning, stop-word filtering, and NLTK-based Lemmatization to reduce words to their base form.

---

## 📊 Model Performance

Trained on a representative stratified sample of **50,000** reviews (balanced class weights applied):

* **Overall Accuracy**: `82.61%`
* **Positive (4-5 Stars) F1-Score**: `0.91`
* **Negative (1-2 Stars) F1-Score**: `0.70`
* **Neutral (3 Stars) F1-Score**: `0.39` (improved from 0.26 via balanced weighting)

---

## 📁 Project Structure

* `train.py`: Preprocessing, TF-IDF vectorization, model training, evaluation metrics logging, and artifact generation.
* `app.py`: Streamlit application layout, metrics indicators, charting configurations, and custom styled predictor components.
* `model.pkl` & `vectorizer.pkl`: Saved machine learning model artifacts.
* `sampled_reviews.csv`: Pre-sampled subset of reviews for fast dashboard loading.
* `requirements.txt`: Python package requirements.
* `.gitignore`: Excludes the 300MB raw dataset (`Reviews.csv`) to stay within GitHub's file limit.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mansi091/Amazon-reviews-sentiment-analysis.git
   cd Amazon-reviews-sentiment-analysis
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```
   *The app will be accessible at `http://localhost:8501`*

4. **(Optional) Re-train the Model**:
   If you have the raw `Reviews.csv` file in your root directory, you can re-train the model by running:
   ```bash
   python train.py
   ```
