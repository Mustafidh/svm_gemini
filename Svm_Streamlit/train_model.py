import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from collections import Counter

from nltk.corpus import stopwords
from PIL import Image 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
kamus_path = os.path.join(BASE_DIR, "data", "kamuskatabaku.xlsx")
positive_path = os.path.join(BASE_DIR, "lexicon", "positive.txt")
negative_path = os.path.join(BASE_DIR, "lexicon", "negative.txt")

# Download stopwords
nltk.download('stopwords')
st.set_page_config(page_title="Analisis Sentimen Gemini", layout="wide")
# ==== Load resource ====
@st.cache_data
def load_resources():
    try:
        kamus_data = pd.read_excel(kamus_file)
        kamus_tidak_baku = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))
    except:
        kamus_tidak_baku = {}
    
    try:
       with open(positive_path, 'r', encoding='utf-8') as f:
    positive_lexicon = set(line.strip() for line in f if line.strip())
    except:
        positive_lexicon = set()
    
    try:
       with open(negative_path, 'r', encoding='utf-8') as f:
    negative_lexicon = set(line.strip() for line in f if line.strip())
    except:
        negative_lexicon = set()
    
    return kamus_tidak_baku, positive_lexicon, negative_lexicon

kamus_tidak_baku, positive_lexicon, negative_lexicon = load_resources()

stop_words = stopwords.words('indonesian')
stemmer = StemmerFactory().create_stemmer()

# ==== Fungsi preprocessing ====
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

def replace_words(text, kamus):
    words = text.split()
    return ' '.join([kamus.get(word, word) for word in words])

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def stem_text(tokens):
    return ' '.join([stemmer.stem(word) for word in tokens])

def determine_sentiment(text):
    text = clean_text(text)
    words = text.split()
    pos = sum(1 for word in words if word in positive_lexicon)
    neg = sum(1 for word in words if word in negative_lexicon)
    return 'Positif' if pos > neg else 'Negatif'

# ==== Fungsi untuk infografik ====
def create_sentiment_distribution_chart(df):
    sentiment_counts = df['sentiment'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=sentiment_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title={
            'text': '📊 Distribusi Sentimen dalam Dataset',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Kategori Sentimen",
        yaxis_title="Jumlah Review",
        template='plotly_white',
        height=400
    )
    
    return fig

def create_sentiment_pie_chart(df):
    sentiment_counts = df['sentiment'].value_counts()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.3,
            marker_colors=['#FF6B6B', '#4ECDC4'],
            textinfo='label+percent+value',
            textfont_size=12
        )
    ])
    
    fig.update_layout(
        title={
            'text': '🥧 Persentase Distribusi Sentimen',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=400
    )
    
    return fig

def create_word_frequency_chart(df):
    positive_text = ' '.join(df[df['sentiment'] == 'Positif']['stemmed'])
    negative_text = ' '.join(df[df['sentiment'] == 'Negatif']['stemmed'])
    
    pos_words = Counter(positive_text.split())
    neg_words = Counter(negative_text.split())
    
    top_pos = dict(pos_words.most_common(15))
    top_neg = dict(neg_words.most_common(15))
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('📈 Kata Tersering - Positif', '📉 Kata Tersering - Negatif'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(
            x=list(top_pos.values()),
            y=list(top_pos.keys()),
            orientation='h',
            marker_color='#4ECDC4',
            name='Positif'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=list(top_neg.values()),
            y=list(top_neg.keys()),
            orientation='h',
            marker_color='#FF6B6B',
            name='Negatif'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title={
            'text': '📝 Analisis Frekuensi Kata Berdasarkan Sentimen',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=600,
        template='plotly_white'
    )
    
    return fig

def create_wordcloud(df, sentiment_type):
    text = ' '.join(df[df['sentiment'] == sentiment_type]['stemmed'])
    
    if text.strip():
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis' if sentiment_type == 'Positif' else 'Reds',
            max_words=100,
            relative_scaling=0.5,
            stopwords=set(stop_words)
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'☁️ Word Cloud - Sentimen {sentiment_type}', fontsize=16, fontweight='bold')
        
        return fig
    else:
        return None

def create_confusion_matrix_heatmap(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted Negatif', 'Predicted Positif'],
        y=['Actual Negatif', 'Actual Positif'],
        colorscale='Blues',
        text=conf_matrix,
        texttemplate="%{text}",
        textfont={"size": 20},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title={
            'text': '🎯 Confusion Matrix - Performa Model',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Prediksi Model",
        yaxis_title="Nilai Aktual",
        height=400
    )
    
    return fig

def create_metrics_dashboard(y_test, y_pred, accuracy):
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Akurasi",
            value=f"{accuracy * 100:.2f}%"
        )
    
    with col2:
        st.metric(
            label="📊 Precision (Avg)",
            value=f"{report['weighted avg']['precision']:.2f}"
        )
    
    with col3:
        st.metric(
            label="🔍 Recall (Avg)",
            value=f"{report['weighted avg']['recall']:.2f}"
        )
    
    with col4:
        st.metric(
            label="⚡ F1-Score (Avg)",
            value=f"{report['weighted avg']['f1-score']:.2f}"
        )

def create_sample_analysis(df):
    st.subheader("🔍 Contoh Analisis Sentimen")
    
    # Ambil beberapa contoh dari setiap kategori
    pos_samples = df[df['sentiment'] == 'Positif'].head(3)
    neg_samples = df[df['sentiment'] == 'Negatif'].head(3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### ✅ Contoh Review Positif")
        for idx, row in pos_samples.iterrows():
            with st.expander(f"Review {idx+1}"):
                st.write(f"**Original:** {row['Review Text'][:100]}...")
                st.write(f"**Processed:** {row['stemmed'][:100]}...")
                st.write(f"**Sentiment:** {row['sentiment']}")
    
    with col2:
        st.write("### ❌ Contoh Review Negatif")
        for idx, row in neg_samples.iterrows():
            with st.expander(f"Review {idx+1}"):
                st.write(f"**Original:** {row['Review Text'][:100]}...")
                st.write(f"**Processed:** {row['stemmed'][:100]}...")
                st.write(f"**Sentiment:** {row['sentiment']}")

# ==== Streamlit UI ====
st.title("🚀 Pelatihan Model SVM untuk Analisis Sentimen Aplikasi Gemini")

# Gambar logo
try:
    image_path = os.path.join(os.path.dirname(__file__), "img", "gemini.png")
    image = Image.open(image_path)

    col1, col2, col3 = st.columns([1, 3, 1])  # kolom tengah lebih lebar
    with col2:
        st.image(image, caption='', use_container_width=True)  # ✅ parameter terbaru
except:
    st.info("📷 Gambar logo tidak ditemukan, melanjutkan tanpa gambar.")

st.markdown("""
### 📋 Deskripsi
Aplikasi ini menggunakan Machine Learning dengan algoritma Support Vector Machine (SVM) untuk menganalisis sentimen review aplikasi Gemini. 
Model akan dilatih untuk mengklasifikasikan review menjadi sentimen **Positif** atau **Negatif**.
""")

st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload File CSV", type=["csv"])

if uploaded_file:
    # Load data
    with st.spinner("⏳ Memproses data..."):
        df = pd.read_csv(uploaded_file)
        
        if 'Review Text' not in df.columns:
            st.error("❌ Kolom 'Review Text' tidak ditemukan dalam file CSV.")
            st.stop()
        
        # Preprocessing
        df.drop_duplicates(subset=['Review Text'], inplace=True)
        df['clean'] = df['Review Text'].astype(str).apply(clean_text)
        df['normalized'] = df['clean'].apply(lambda x: replace_words(x, kamus_tidak_baku))
        df['token'] = df['normalized'].apply(str.split)
        df['stopword'] = df['token'].apply(remove_stopwords)
        df['stemmed'] = df['stopword'].apply(stem_text)
        df['sentiment'] = df['stemmed'].apply(determine_sentiment)
    
    # Dataset Overview
    st.subheader("📊 Overview Dataset")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📝 Total Reviews", len(df))
    with col2:
        st.metric("✅ Positif", len(df[df['sentiment'] == 'Positif']))
    with col3:
        st.metric("❌ Negatif", len(df[df['sentiment'] == 'Negatif']))
    
    # Infografik Section
    st.markdown("---")
    st.header("📈 Infografik Analisis Sentimen")
    
    # Row 1: Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = create_sentiment_distribution_chart(df)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        fig_pie = create_sentiment_pie_chart(df)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Row 2: Word frequency analysis
    st.plotly_chart(create_word_frequency_chart(df), use_container_width=True)
    
    # Row 3: Word clouds
    st.subheader("☁️ Word Cloud Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pos = create_wordcloud(df, 'Positif')
        if fig_pos:
            st.pyplot(fig_pos)
    
    with col2:
        fig_neg = create_wordcloud(df, 'Negatif')
        if fig_neg:
            st.pyplot(fig_neg)
    
    # Sample analysis
    create_sample_analysis(df)
    
    # Model Training
    st.markdown("---")
    st.header("🤖 Pelatihan Model SVM")
    
    with st.spinner("🔄 Melatih model..."):
        # Prepare data
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = tfidf.fit_transform(df['stemmed'])
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        svm = LinearSVC(random_state=42)
        svm.fit(X_train, y_train)
        
        # Predictions
        y_pred = svm.predict(X_test)
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
    
    # Results Section
    st.markdown("---")
    st.header("📊 Hasil Evaluasi Model")
    
    # Metrics dashboard
    create_metrics_dashboard(y_test, y_pred, accuracy)
    
    # Confusion matrix
    st.plotly_chart(create_confusion_matrix_heatmap(y_test, y_pred), use_container_width=True)
    
    # Detailed classification report
    st.subheader("📋 Laporan Klasifikasi Detail")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Format the dataframe for better display
    report_df = report_df.round(3)
    st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
    
    # Feature importance (top TF-IDF features)
    st.subheader("🔍 Kata-kata Penting dalam Model")
    feature_names = tfidf.get_feature_names_out()
    importance_scores = np.abs(svm.coef_[0])
    
    # Get top features
    top_features = sorted(zip(feature_names, importance_scores), key=lambda x: x[1], reverse=True)[:20]
    
    feature_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
    
    fig_features = px.bar(
        feature_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='🏆 Top 20 Fitur Penting dalam Model SVM',
        color='Importance',
        color_continuous_scale='viridis'
    )
    fig_features.update_layout(height=600)
    st.plotly_chart(fig_features, use_container_width=True)
    
    # Save model
    st.markdown("---")
    st.subheader("💾 Simpan Model")
    
    if st.button("💾 Simpan Model dan Vectorizer"):
        with st.spinner("💾 Menyimpan model..."):
            os.makedirs("models", exist_ok=True)
            joblib.dump(svm, "models/svm_model.pkl")
            joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
            
            # Save model info
            model_info = {
                'accuracy': accuracy,
                'total_samples': len(df),
                'features_count': X.shape[1],
                'positive_samples': len(df[df['sentiment'] == 'Positif']),
                'negative_samples': len(df[df['sentiment'] == 'Negatif'])
            }
            
            import json
            with open("models/model_info.json", "w") as f:
                json.dump(model_info, f)
            
            st.success("✅ Model, TF-IDF vectorizer, dan informasi model berhasil disimpan!")
    
    # Download processed data
    st.subheader("📥 Download Data Hasil Preprocessing")
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV Hasil Preprocessing",
        data=csv,
        file_name="processed_sentiment_data.csv",
        mime="text/csv"
    )

else:
    st.info("📂 Silakan upload file CSV berisi kolom 'Review Text' untuk memulai analisis.")
    
    # Show example format
    st.subheader("📋 Format File CSV yang Dibutuhkan")
    example_df = pd.DataFrame({
        'Review Text': [
            'Aplikasi ini sangat bagus dan mudah digunakan',
            'Saya tidak suka dengan aplikasi ini, terlalu lambat',
            'Gemini membantu saya dalam belajar, recommended!',
            'Aplikasi jelek, banyak bug dan error'
        ]
    })
    st.dataframe(example_df, use_container_width=True)
    
    st.write("**Catatan:** File CSV harus memiliki kolom 'Review Text' yang berisi teks review yang akan dianalisis.")
