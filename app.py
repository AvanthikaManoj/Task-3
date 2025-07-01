import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


samples = [
    ("I love this product!", "positive"),
    ("Amazing experience, very happy", "positive"),
    ("Worst service ever", "negative"),
    ("I hate this item", "negative"),
    ("It's okay, not too great", "neutral"),
    ("Nothing special, just average", "neutral")
]

texts, labels = zip(*samples)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)


st.title("💬 Social Media Sentiment Analysis (Simple Demo)")

user_input = st.text_area("Type a comment here:")

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.error("Please enter some text to analyze.")
    else:
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        st.success(f"🎯 Sentiment: **{prediction.capitalize()}**")
