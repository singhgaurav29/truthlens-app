
import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords

# NLTK resources
nltk.download('stopwords', quiet=True)

# Page config
st.set_page_config(page_title='TruthLens', page_icon='🔍', layout='centered')

# Styling (PDF ke CSS patterns ke hisab se)
st.markdown('''
<style>
.verdict-fake { background: #c1121f; color:white; padding:18px; border-radius:8px; text-align:center; font-size:26px; font-weight:bold; }
.verdict-real { background: #1b6b3a; color:white; padding:18px; border-radius:8px; text-align:center; font-size:26px; font-weight:bold; }
.signal-box { background:#f0f4f8; border-left: 4px solid #1a1a2e; padding:12px 16px; margin:8px 0; border-radius: 4px; }
</style>
''', unsafe_allow_html=True)

# Load saved model and vectorizer
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vec = pickle.load(f)
    return model, vec

model, vectorizer = load_model()

# Prediction function
def predict_with_explanation(headline):
    vec_input = vectorizer.transform([headline])
    prediction = model.predict(vec_input)[0]
    proba = model.predict_proba(vec_input)[0]
    confidence = max(proba) * 100
    
    feature_names = vectorizer.get_feature_names_out()
    dense = vec_input.toarray()[0]
    fake_idx = list(model.classes_).index('FAKE')
    
    # Probability ke basis par signals nikalna
    word_scores = dense * model.feature_log_prob_[fake_idx]
    top_indices = word_scores.argsort()[-3:][::-1]
    top_signals = [feature_names[i] for i in top_indices if dense[i] > 0]
    
    return prediction, confidence, top_signals

# UI Part
st.title('🔍 TruthLens')
st.subheader('AI-Powered Fake News Detector')
st.caption('Built from scratch over 8 weeks — by a Class 8 student')
st.divider()

headline_input = st.text_area('Paste a news headline here:', height=100)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyse = st.button('ANALYSE THIS HEADLINE', type='primary', use_container_width=True)

if analyse:
    if headline_input:
        pred, conf, signals = predict_with_explanation(headline_input)
        if pred == 'FAKE':
            st.markdown(f'<div class="verdict-fake">LIKELY FAKE ({conf:.0f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-real">LIKELY REAL ({conf:.0f}%)</div>', unsafe_allow_html=True)
        
        if signals:
            st.write("Top signals found:")
            for s in signals:
                st.markdown(f'<div class="signal-box">Word: <b>{s}</b></div>', unsafe_allow_html=True)
