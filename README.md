# 📧 Email Spam Detector using Machine Learning

A lightweight email spam detection system using Natural Language Processing (NLP), TF-IDF vectorization, and a Naive Bayes classifier. The project includes a simple Streamlit web app to classify emails as Spam or Ham.

---

## 🗂 Table of Contents
- [🎯 Objectives](#-objectives)
- [📁 Project Structure](#-project-structure)
- [📦 Installation](#-installation)
- [⚙ How to Use](#-how-to-use)
  - [Train the Model](#train-the-model)
  - [Run the App](#run-the-app)
- [🧠 Model and Techniques](#-model-and-techniques)
- [📊 Evaluation Metrics](#-evaluation-metrics)
- [📈 Sample Output](#-sample-output)
- [🚀 Future Enhancements](#-future-enhancements)
- [👨‍💻 Author](#-author)

---

## 🎯 Objectives
- Automate email classification to filter out spam.
- Build a simple but effective text classification pipeline.
- Provide a real-time interface for spam prediction.
- Export and reuse trained models (.pkl).

---

## 📁 Project Structure

email-spam-detector/
│
├── data/
│   └── enron_spam_data.csv ← Dataset (emails + labels)
├── spam_model.py ← Model training script
├── app.py ← Streamlit web app
├── spam_classifier.pkl ← Trained Naive Bayes model
├── tfidf_vectorizer.pkl ← TF-IDF vectorizer
├── requirements.txt ← Python dependencies
└── README.md ← Project documentation


---

## 📦 Installation

Clone the repository and install the dependencies:
bash
git clone https://github.com/your-username/email-spam-detector.git
cd email-spam-detector
pip install -r requirements.txt


Or manually install:
bash
pip install pandas scikit-learn streamlit


---

## ⚙ How to Use

### 📌 Prerequisites
Ensure your dataset is present at:
data/enron_spam_data.csv
with columns:
- Message: the text of the email
- Spam/Ham: the label (spam or ham)

### 🧠 Train the Model
Run the training script:
bash
python spam_model.py

This will:
- Preprocess the dataset
- Train a Multinomial Naive Bayes model
- Save spam_classifier.pkl and tfidf_vectorizer.pkl
- Display evaluation metrics (Accuracy, Classification Report, Confusion Matrix)

### 💻 Run the App
Start the Streamlit interface:
bash
streamlit run app.py

Then open the link in your browser to test your own email messages.

---

## 🧠 Model and Techniques
- *Classifier*: Multinomial Naive Bayes
- *Text Vectorization*: TF-IDF (max_features=5000)
- *Data Split*: 80% Train / 20% Test

### Preprocessing:
- Lowercasing
- Removing emails, URLs, punctuation, numbers
- Whitespace normalization

---

## 📊 Evaluation Metrics
Displayed after training:
- Accuracy
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix

---

## 📈 Sample Output

Accuracy: 97.85%
Classification Report:
              precision    recall  f1-score   support
         Ham       0.98      0.99      0.98       965
        Spam       0.97      0.96      0.96       700
    accuracy                           0.98      1665
   macro avg       0.97      0.97      0.97      1665
weighted avg       0.98      0.98      0.98      1665


---

## 🚀 Future Enhancements
- Add other classifiers (SVM, Logistic Regression)
- Deploy to Streamlit Cloud / Hugging Face
- Add batch prediction via file upload
- Add visualization for confusion matrix in the UI
- Improve preprocessing with lemmatization
