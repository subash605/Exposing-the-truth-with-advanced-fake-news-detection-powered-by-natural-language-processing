from google.colab import files

import io

import pandas as pd

# Upload file

uploaded = files.upload()

filename = next(iter(uploaded))

# Read CSV with explicit encoding

df = pd.read_csv(io.BytesIO(uploaded[filename]), encoding="latin-1")

import numpy as np

import pandas as pd

df = pd.read_csv("fake_news_dataset.csv")

df.head(5)

import pandas as pd

import numpy as np

import re

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, 

confusion_matrix

import gradio as gr

# Display Dataset info

print(df.head())

print(df.info())

# Data Preprocessing

def clean_text(text):

text = text.lower()

text = re.sub(r'\[.*?\]', '', text)

text = re.sub(r'http\S+|www.\S+', '', text)

text = re.sub(r'<.*?>+', '', text)

text = re.sub(r'[^a-zA-Z\s]', '', text)

text = re.sub(r'\n', ' ', text)

text = re.sub(r'\w*\d\w*', '', text)

return text

df['test'] = df['text'].apply(clean_text)

# Splitting Data

X = df['text']

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 

random_state=42)

# Feature Extraction with TF-IDF

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Model Building - Logistic Regression

model = LogisticRegression()

model.fit(X_train_tfidf, y_train)

# Model Evaluation

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization

plt.figure(figsize=(5,4))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()

# Deploy with Gradio

def predict_fake_news(input_text):

input_cleaned = clean_text(input_text)

input_vector = tfidf_vectorizer.transform([input_cleaned])

prediction = model.predict(input_vector)[0]

result = 'Real News' if prediction == 1 else 'Fake News'

return result

interface = gr.Interface(fn=predict_fake_news,

inputs=gr.Textbox(lines=5, placeholder="Enter news article text 

here..."),

outputs="text",

title="Fake News Detection",

description="Enter a news article text to check if it's real or fake 

using NLP and ML.")

interface.launch()
