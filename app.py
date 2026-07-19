import streamlit as st
import pickle
import re

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("📰 Fake News Detector")
st.write("Koi bhi news paste karo, model bataega Fake hai ya Real.")

news_input = st.text_area("News Text yahan paste karo:", height=200)

if st.button("Check Karo 🔍"):
    if news_input.strip() == "":
        st.warning("Pehle kuch text daalo!")
    else:
        cleaned = clean_text(news_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max() * 100
        if prediction == 1:
            st.success(f"✅ REAL lagti hai (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"❌ FAKE lagti hai (Confidence: {confidence:.2f}%)")
            