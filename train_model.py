import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pickle
import re
import email

# NLTK imports
import nltk
nltk.data.path.append("nltk_data")  # استخدام بيانات NLTK من مجلد محلي
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Function to preprocess email text
def preprocess_email(text):
    """Preprocess the text by removing headers, links, tokenizing, and lemmatizing."""

    # Decode email if in bytes
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')

    # Parse email if it's a string
    msg = email.message_from_string(text) if isinstance(text, str) else text

    # Extract email body
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                text = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                break
    else:
        text = msg.get_payload(decode=True).decode('utf-8', errors='ignore') if msg.get_payload() else text

    # Clean text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Load and clean dataset
def load_enron_data(data_path='data/enron_spam_data.csv'):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Message'])
    df['cleaned_text'] = df['Message'].apply(preprocess_email)
    df['label'] = df['Spam/Ham'].map({'spam': 1, 'ham': 0})
    return df[['cleaned_text', 'label']]

# Train and save model
def train_and_save_model():
    df = load_enron_data()
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    with open('spam_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    y_pred = model.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
