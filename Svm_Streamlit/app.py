import streamlit as st
import pandas as pd
import re
import joblib
import os
import nltk

from PIL import Image 
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download stopwords (sekali saja)
nltk.download('stopwords')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Load Resource ===
model_path = os.path.join(BASE_DIR, "models", "svm_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
kamus_path = os.path.join(BASE_DIR, "data", "kamuskatabaku.xlsx")
positive_path = os.path.join(BASE_DIR, "lexicon", "positive.txt")
negative_path = os.path.join(BASE_DIR, "lexicon", "negative.txt")

# Cek model
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Model belum tersedia. Jalankan train_model.py terlebih dahulu untuk membuat model.")
    st.stop()

# Load model dan vectorizer
svm = joblib.load(model_path)
tfidf = joblib.load(vectorizer_path)

# Load kamus tidak baku
kamus_df = pd.read_excel(kamus_path)
kamus_tidak_baku = dict(zip(kamus_df['tidak_baku'], kamus_df['kata_baku']))

# Load lexicon
with open(positive_path, 'r', encoding='utf-8') as f:
    positive_lexicon = set(line.strip() for line in f if line.strip())
with open(negative_path, 'r', encoding='utf-8') as f:
    negative_lexicon = set(line.strip() for line in f if line.strip())

# Preprocessing tools
stop_words = stopwords.words('indonesian')
stemmer = StemmerFactory().create_stemmer()

def remove_emoji(text):
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"u"\U0001FA70-\U0001FAFF"
        u"\U0001F004-\U0001F0CF"u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', str(text))

def remove_symbols(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text))

def remove_numbers(text):
    return re.sub(r'\d+', '', str(text))

def case_folding(text):
    return text.lower()

def replace_taboo_words(text):
    words = text.split()
    return ' '.join([kamus_tidak_baku[word] if word in kamus_tidak_baku else word for word in words])

def tokenize(text):
    return text.split()

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def stem_text(tokens):
    return ' '.join([stemmer.stem(word) for word in tokens])

def full_preprocess(text):
    text = remove_emoji(text)
    text = remove_symbols(text)
    text = remove_numbers(text)
    text = case_folding(text)
    text = replace_taboo_words(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return stem_text(tokens)

def classify_sentiment(text):
    cleaned = full_preprocess(text)
    vector = tfidf.transform([cleaned])
    prediction = svm.predict(vector)[0]
    return prediction

# CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        padding: 2rem;
        max-width: 800px;
        margin: auto;
    }

    .title {
        color: #003566;
        font-weight: 700;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        color: #6c757d;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 1.8rem;
    }

    .result-box {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 1rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .positive {
        background-color: #d1e7dd;
        color: #0f5132;
        border: 1px solid #badbcc;
    }

    .negative {
        background-color: #f8d7da;
        color: #842029;
        border: 1px solid #f5c2c7;
    }

    .stTextArea textarea {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #d6d6d6;
        font-size: 1rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #0077b6, #00b4d8);
        color: white !important;
        fant-family: 'Poppins', sans-serif;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        border: none;
        font-size: 1rem;
        width: 100%;
        transition: transform 0.2s ease;
    }

    .stButton > button:active {
        transform: scale(0.98);
    }

    .gemini-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 120px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Kontainer utama
st.markdown('<div class="main">', unsafe_allow_html=True)

# Judul
st.markdown('<h1 class="title">Analisis Sentimen Aplikasi Gemini</h1>', unsafe_allow_html=True)

# Gambar logo
image_path = os.path.join("img", "gemini.png")
image = Image.open(image_path)
st.image(image, caption='')

# Subjudul
st.markdown('<p class="subtitle">Masukkan kalimat dan sistem akan mengklasifikasikan apakah sentimennya <strong>positif</strong> atau <strong>negatif</strong>.</p>', unsafe_allow_html=True)

# Input teks
user_input = st.text_area("Masukkan ulasan Anda di bawah ini:")
if st.button("Klasifikasikan"):
    if user_input:
        pred = classify_sentiment(user_input)
        st.markdown("**Hasil Analisis Sentimen:**")
        if pred.lower() == "positif":
            st.success("✅ Sentimen: **Positif**")
        else:
            st.error("❌ Sentimen: **Negatif**")
    else:
        st.warning("Mohon masukkan teks terlebih dahulu.")

st.markdown('</div>', unsafe_allow_html=True)
