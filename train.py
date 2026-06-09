import pandas as pd
import numpy as np
import re
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    start_time = time.time()
    csv_path = "Reviews.csv"
    print("Loading dataset...")
    cols = ['Score', 'Summary', 'Text', 'HelpfulnessNumerator', 'HelpfulnessDenominator']
    df = pd.read_csv(csv_path, usecols=cols)
    print(f"Loaded {len(df)} rows.")
    sample_eda = df.sample(n=10000, random_state=42)
    sample_eda.to_csv("sampled_reviews.csv", index=False)
    print("Saved sampled_reviews.csv for Streamlit EDA.")
    def map_sentiment(score):
        if score <= 2:
            return 0
        elif score == 3:
            return 1
        else:
            return 2
    df['Sentiment'] = df['Score'].apply(map_sentiment)
    df['Combined_Text'] = df['Summary'].fillna('') + ' ' + df['Text'].fillna('')
    train_size = min(50000, len(df))
    if train_size < len(df):
        _, train_sample = train_test_split(
            df,
            test_size=train_size,
            random_state=42,
            stratify=df['Sentiment']
        )
    else:
        train_sample = df.copy()
    print(f"Training subset size: {len(train_sample)} rows.")
    print("Preprocessing text...")
    train_sample['Cleaned_Text'] = train_sample['Combined_Text'].apply(clean_text)
    train_sample = train_sample[train_sample['Cleaned_Text'] != ""]
    X = train_sample['Cleaned_Text']
    y = train_sample['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train_tfidf, y_train)
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    target_names = ['Negative', 'Neutral', 'Positive']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Saving model and vectorizer...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Done! Total time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()

