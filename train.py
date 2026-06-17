import pandas as pd
import numpy as np
import re
import pickle
import time
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def main():
    import os
    from joblib import Parallel, delayed

    start_time = time.time()
    csv_path = "Reviews.csv"
    fallback_path = "sampled_reviews.csv"
    
    if not os.path.exists(csv_path):
        if os.path.exists(fallback_path):
            print(f"Warning: '{csv_path}' not found. Falling back to training on '{fallback_path}'...")
            df = pd.read_csv(fallback_path)
            cols = ['Score', 'Summary', 'Text', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time']
            for col in cols:
                if col not in df.columns:
                    df[col] = np.nan
        else:
            print(f"Error: Neither '{csv_path}' nor '{fallback_path}' was found. Cannot train.")
            return
    else:
        print("Loading dataset...")
        cols = ['Score', 'Summary', 'Text', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time']
        df = pd.read_csv(csv_path, usecols=cols)
        print(f"Loaded {len(df)} rows.")
        
        print("Generating sampled_reviews.csv for Streamlit EDA...")
        sample_eda = df.sample(n=min(10000, len(df)), random_state=42)
        print("Preprocessing EDA sample...")
        sample_eda['Combined_Text'] = sample_eda['Summary'].fillna('') + ' ' + sample_eda['Text'].fillna('')
        sample_eda['Cleaned_Text'] = Parallel(n_jobs=-1, batch_size=500)(
            delayed(clean_text)(t) for t in sample_eda['Combined_Text']
        )
        sample_eda = sample_eda.drop(columns=['Combined_Text'])
        sample_eda.to_csv(fallback_path, index=False)
        print("Saved sampled_reviews.csv.")

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
    
    if 'Cleaned_Text' in train_sample.columns and train_sample['Cleaned_Text'].notna().sum() > 0:
        print("Using pre-cleaned text...")
        train_sample['Cleaned_Text'] = train_sample['Cleaned_Text'].fillna('')
    else:
        print("Cleaning text in parallel...")
        train_sample['Cleaned_Text'] = Parallel(n_jobs=-1, batch_size=500)(
            delayed(clean_text)(t) for t in train_sample['Combined_Text']
        )
        if not os.path.exists(csv_path):
            print("Saving pre-cleaned text to sampled_reviews.csv for future runs...")
            df['Cleaned_Text'] = train_sample['Cleaned_Text']
            df.to_csv(fallback_path, index=False)
        
    train_sample = train_sample[train_sample['Cleaned_Text'].str.strip() != ""]
    X = train_sample['Cleaned_Text']
    y = train_sample['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced'),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced', n_jobs=-1)
    }
    
    comparison_metrics = {}
    
    for name, clf in models.items():
        print(f"Training {name} model...")
        model_start = time.time()
        clf.fit(X_train_tfidf, y_train)
        print(f"{name} trained in {time.time() - model_start:.2f} seconds.")
        
        y_pred = clf.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
        macro_f1 = report['macro avg']['f1-score']
        
        print(f"{name} Accuracy: {acc:.4f} | Macro F1-Score: {macro_f1:.4f}")
        
        comparison_metrics[name] = {
            "Accuracy": acc,
            "Macro_F1": macro_f1,
            "report": report
        }
        
    print("Saving comparison metrics to model_comparison.json...")
    with open("model_comparison.json", "w") as f:
        json.dump(comparison_metrics, f, indent=4)
        
    print("Saving models and vectorizer...")
    with open("model_logistic.pkl", "wb") as f:
        pickle.dump(models["Logistic Regression"], f)
    with open("model_naive_bayes.pkl", "wb") as f:
        pickle.dump(models["Naive Bayes"], f)
    with open("model_random_forest.pkl", "wb") as f:
        pickle.dump(models["Random Forest"], f)
    with open("model.pkl", "wb") as f:
        pickle.dump(models["Logistic Regression"], f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("Model files successfully serialized and saved.")
    print(f"Done! Total time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
