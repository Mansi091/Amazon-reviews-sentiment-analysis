# Amazon Reviews Sentiment Analysis & Dashboard

An interactive Streamlit web dashboard and multi-model machine learning pipeline for predicting and exploring product review sentiment.

---

## 🚀 Key Pros

* **Multi-Model Support**: Train, evaluate, and compare three classifiers: **Logistic Regression**, **Naive Bayes**, and **Random Forest**.
* **Sub-Second Performance**: Pre-cleaned data sampling skips text preprocessing overhead, reducing re-training times to **1.2 seconds** and rendering word clouds instantly.
* **Interactive Visualizations**: Sleek, responsive, dark-themed charts powered by **Plotly Express** (ratings distribution, sentiment breakdown, trends, and prediction confidence).
* **Developer Friendly**: Robust script falls back to local data if the large raw `Reviews.csv` is missing, ensuring immediate out-of-the-box runnability.
* **Custom Data Upload**: Upload custom CSV dataset files directly from the dashboard to run sentiment predictions and plots on-the-fly.

---

## 📊 Model Performance

| Model | Accuracy | Macro F1-Score |
|---|---|---|
| **Logistic Regression** | **81.15%** | **0.6130** |
| **Multinomial Naive Bayes** | 79.00% | 0.3170 |
| **Random Forest** | 75.20% | 0.5678 |

---

## 🛠️ Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

3. **Re-train the Models**:
   ```bash
   python train.py
   ```
