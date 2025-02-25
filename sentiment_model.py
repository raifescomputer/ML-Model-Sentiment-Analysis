import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

df = pd.read_csv('data/IMDB Dataset.csv')

stop_words_list = list(ENGLISH_STOP_WORDS)
vectorizer = TfidfVectorizer(stop_words=stop_words_list)
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

new_review = ["The movie was okay."]
new_review_vectorized = vectorizer.transform(new_review)
prediction = model.predict(new_review_vectorized)
print(f"Prediction for new review: {prediction}")
