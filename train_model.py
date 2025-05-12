import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import email
from sklearn.metrics import classification_report, accuracy_score
import nltk
nltk.download('stopwords')
nltk.download('punkt') 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.data.path.append("nltk_data")
# Function to preprocess email text (cleaning, tokenizing, lemmatizing)
def preprocess_email(text):
    """Preprocess the text by removing headers, links, tokenizing, and lemmatizing."""
    
    # Decode the email if it's in bytes
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')

    # Parse the email if it's a string
    msg = email.message_from_string(text) if isinstance(text, str) else text

    # Extract text if the email is multipart
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                text = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                break
    else:
        text = msg.get_payload(decode=True).decode('utf-8', errors='ignore') if msg.get_payload() else text

    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabetic characters
    text = text.lower()  # Convert text to lowercase

    # Tokenization: Split the text into words
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization: Reduce words to their base form
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Function to load and clean the Enron email dataset
def load_enron_data(data_path='data\enron_spam_data.csv'):
    """Load the Enron email dataset and clean the text."""
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Message'])  # Drop rows with missing email messages
    df['cleaned_text'] = df['Message'].apply(preprocess_email)  # Preprocess each email message
    df['label'] = df['Spam/Ham'].map({'spam': 1, 'ham': 0})  # Convert spam/ham labels to 1/0
    return df[['cleaned_text', 'label']]  # Return the cleaned text and labels

# Function to train the model and save it
def train_and_save_model():
    """Train the spam classifier model and save the trained model and vectorizer."""
    # Load and clean the Enron dataset
    df = load_enron_data()
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)
    
    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train a Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Save the trained model and vectorizer using pickle
    with open('spam_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Evaluate model performance
    y_pred = model.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))  # Accuracy metric
    print("Classification Report:\n", classification_report(y_test, y_pred))  # Detailed performance report
    
    print("Model and vectorizer saved successfully.")

# Run the training process if this script is executed directly
if __name__ == "__main__":
    train_and_save_model()
