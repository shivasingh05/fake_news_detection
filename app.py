import streamlit as st
import pickle
import re

# Page settings
st.set_page_config(page_title="NewsCheck AI", page_icon="🛡️", layout="centered")

# ---------- Custom Styling ----------
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #9333EA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #666;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
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

# ---------- Input Section ----------
st.subheader("📋 Paste News Article Below")
news_input = st.text_area("", height=180, placeholder="Paste any news headline or article text here...")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    check = st.button("🔍 Analyze News", use_container_width=True)

st.divider()

# ---------- Result ----------
if check:
    if news_input.strip() == "":
        st.warning("⚠️ Please paste some news text first!")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(news_input)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]
            confidence = model.predict_proba(vec).max() * 100

        if prediction == 1:
            result_text = "This news looks REAL"
            st.success(f"### ✅ {result_text}")
            st.metric("Confidence Score", f"{confidence:.1f}%")
        else:
            result_text = "This news looks FAKE"
            st.error(f"### ❌ {result_text}")
            st.metric("Confidence Score", f"{confidence:.1f}%")

        speech_text = f"{result_text}. Confidence {confidence:.0f} percent."
        st.components.v1.html(f"""
            <script>
            var msg = new SpeechSynthesisUtterance("{speech_text}");
            window.speechSynthesis.speak(msg);
            </script>
        """, height=0)