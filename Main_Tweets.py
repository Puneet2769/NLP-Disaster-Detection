import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def main():
    # Load data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # Fill missing text
    train["text"] = train["text"].fillna("")
    test["text"] = test["text"].fillna("")

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    X = vectorizer.fit_transform(train["text"])
    X_test = vectorizer.transform(test["text"])
    y = train["target"]

    # Train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Model
    model = LogisticRegression(max_iter=2000)
    model.fit(X_tr, y_tr)

    # Validation score
    preds_val = model.predict(X_val)
    print("Validation F1:", f1_score(y_val, preds_val))

    # Train on full data
    model.fit(X, y)

    # Predict test
    preds_test = model.predict(X_test)

    # Create submission
    sub = pd.DataFrame({
        "id": test["id"],
        "target": preds_test
    })

    sub.to_csv("submission_simple_nlp.csv", index=False)
    print("Saved submission_simple_nlp.csv")

if __name__ == "__main__":
    main()
