import streamlit as st
import pickle
import os
import re
import string

# Text preprocessing function (نفس اللي في training)
def simple_preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text if text.strip() else ""

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open('spam_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

# Streamlit UI
def main():
    st.set_page_config(page_title="Email Spam Detector", layout="centered")
    st.title("📧 Email Spam Detector")
    st.write("Enter an email message below to check if it's spam.")

    model, vectorizer = load_model()

    email_input = st.text_area("Email Message", height=200)

    if st.button("Classify"):
        if email_input.strip() == "":
            st.warning("Please enter a valid email message.")
        else:
            processed_text = simple_preprocess(email_input)
            vector_input = vectorizer.transform([processed_text])
            prediction = model.predict(vector_input)[0]
            prob = model.predict_proba(vector_input)[0]

            if prediction == 1:
                st.error(f"🚫 Spam Detected ({prob[1]*100:.2f}% confidence)")
            else:
                st.success(f"✅ Not Spam ({prob[0]*100:.2f}% confidence)")

if __name__ == "__main__":
    main()