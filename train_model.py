import re
import string
import pandas as pd
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def simple_preprocess(text):
    try:
        text = str(text).lower()
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        return text if text.strip() else ""
    except:
        return ""

def preprocess_email(email_text):
    return simple_preprocess(email_text)

def train_spam_model(csv_path='data/enron_spam_data.csv'):
    model_path = 'spam_classifier.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'

    print("📥 Loading dataset...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Message'])
    df['processed_text'] = df['Message'].apply(preprocess_email)
    df['label'] = df['Spam/Ham'].map({'spam': 1, 'ham': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    print("✅ Model trained.")

    # Evaluation
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n📊 Accuracy: {acc * 100:.2f}%")
    print("\n📝 Classification Report:\n", report)
    print("\n🔢 Confusion Matrix:\n", cm)

    # Save model & vectorizer
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    print("\n💾 Model and vectorizer saved.")

if __name__ == "__main__":
    train_spam_model()
