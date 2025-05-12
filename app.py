import streamlit as st
import pickle
from train_model import preprocess_email
import nltk
nltk.data.path.append("nltk_data")
import nltk
nltk.download('stopwords')
nltk.download('punkt') 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
def load_model():
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
    cleaned_text = preprocess_email(email_text)
    features = vectorizer.transform([cleaned_text])
    prediction = model.predict(features)[0]
    return 'Spam' if prediction == 1 else 'Ham'

# Streamlit Interface
def streamlit_app():
    st.title("Email Spam Detector")
    st.write("Enter the email content to classify it as Spam or Ham")
    
    model, vectorizer = load_model()

    if model is None or vectorizer is None:
        return

    email_text = st.text_area("Email Content", height=200, placeholder="Paste your email text here...")

    if st.button("Classify"):
        if email_text:
            result = classify_email(email_text, model, vectorizer)
            st.success(f"Classification: **{result}**")
        else:
            st.warning("Please enter email text.")

if __name__ == "__main__":
    streamlit_app()
