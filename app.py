import streamlit as st
import pickle
from train_model import preprocess_email
import nltk

# Ensure NLTK data path is set
nltk.data.path.append("nltk_data")

# Download necessary NLTK resources (only if not already downloaded)
def download_nltk_resources():
    """
    Download all required NLTK resources.
    """
    # List of resources to download
    resources = [
        'stopwords',  # For removing stopwords
        'punkt',      # For tokenization
        'wordnet',    # For lemmatization
        'omw-1.4'     # Open Multilingual Wordnet (required for WordNet)
    ]
    
    for resource in resources:
        try:
            # Try to find the resource
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            # If not found, download it
            try:
                nltk.download(resource)
                print(f"Successfully downloaded {resource}")
            except Exception as e:
                print(f"Error downloading {resource}: {e}")

# Download resources when the script is imported
download_nltk_resources()

def load_model():
    """
    Load the pre-trained spam classification model and TF-IDF vectorizer.
    
    Returns:
        tuple: Loaded model and vectorizer, or (None, None) if files not found
    """
    try:
        with open('spam_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first using train_model.py.")
        return None, None

def classify_email(email_text, model, vectorizer):
    """
    Classify an email as Spam or Ham.
    
    Args:
        email_text (str): The email content to classify
        model: Trained classification model
        vectorizer: TF-IDF vectorizer
    
    Returns:
        str: 'Spam' or 'Ham' classification
    """
    cleaned_text = preprocess_email(email_text)
    features = vectorizer.transform([cleaned_text])
    prediction = model.predict(features)[0]
    return 'Spam' if prediction == 1 else 'Ham'

def streamlit_app():
    """
    Create the Streamlit user interface for email spam detection.
    """
    st.title("Email Spam Detector")
    st.write("Enter the email content to classify it as Spam or Ham")
    
    # Load the model and vectorizer
    model, vectorizer = load_model()

    # Exit if model loading failed
    if model is None or vectorizer is None:
        return

    # Text input for email content
    email_text = st.text_area(
        "Email Content", 
        height=200, 
        placeholder="Paste your email text here..."
    )

    # Classification button
    if st.button("Classify"):
        if email_text:
            result = classify_email(email_text, model, vectorizer)
            st.success(f"Classification: **{result}**")
        else:
            st.warning("Please enter email text.")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()