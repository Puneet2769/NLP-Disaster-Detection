# 📝 Disaster Tweet Classifier  
### Logistic Regression + TF-IDF for the Kaggle “NLP Getting Started” Challenge

A simple and effective NLP pipeline that classifies tweets as **real disaster** or **not disaster**.  
This project is built for the Kaggle competition **“Real or Not? Disaster Tweets”** and includes the full workflow from preprocessing to submission.

---

## 📘 Competition  
**Kaggle:** https://www.kaggle.com/competitions/nlp-getting-started

**Dataset Overview:**  
- `train.csv` → tweets + labels (`target`: 1 = disaster, 0 = not disaster)  
- `test.csv` → unlabeled tweets  
- Columns: `id`, `keyword`, `location`, `text`  
- Dataset not included due to Kaggle rules  

---

## ⚙️ What This Project Does  

**Text Preprocessing**  
- Loads train/test CSVs  
- Fills missing text fields with empty strings  
- Uses **TF-IDF** to convert text to numerical vectors  
- Keeps `max_features=20000` for balanced performance  

**Model Training**  
- Train/validation split: 80% / 20%  
- Trains a `LogisticRegression` model (`max_iter=2000`)  
- Evaluates using **F1 score**  
- Retrains on full dataset for final predictions  

**Output**  
- Generates a **Kaggle-ready** submission file  
- Saved as: `submission_simple_nlp.csv`

---

## 🚀 How to Run  

Make sure the working folder contains:

train.csv
test.csv
nlp_tweet_classifier.py
requirements.txt (optional)

Run the script:

python nlp_tweet_classifier.py

yaml
Copy code

The script will:

✔ Load and clean the text  
✔ Train the model  
✔ Print validation F1 score  
✔ Generate `submission_simple_nlp.csv`

---

## 🧠 Model Details  

- **Vectorizer:** `TfidfVectorizer` (English stop-words, 20k features)  
- **Classifier:** `LogisticRegression`  
- **Metric:** F1 score  
- Strong baseline for short text classification  
- Fast and interpretable  

---

## 📁 Repository Structure  

├── nlp_tweet_classifier.py # main training + inference script
├── submission_simple_nlp.csv # generated submission
├── requirements.txt # optional
└── README.md


---

## 👤 Author  
**Puneet Poddar**  
Kaggle Profile: (https://www.kaggle.com/puneet2769)

  - Character-level features  
  - SVM or LinearSVC  
  - Deep learning models (LSTM, BERT, etc.)  
