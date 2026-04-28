import streamlit as st
import pandas as pd
import joblib  # Pickle ki jagah stable loading ke liye [cite: 551]
import re
import nltk
from nltk.corpus import stopwords

# NLTK resources setup [cite: 557]
nltk.download('stopwords', quiet=True)

# 1. Page Configuration [cite: 559]
st.set_page_config(page_title='TruthLens', page_icon='🔍', layout='centered')

# 2. Custom Styling (Teacher Guide CSS) [cite: 561, 562, 564, 565, 566]
st.markdown('''
<style>
    .verdict-fake { background: #c1121f; color:white; padding:18px; border-radius:8px; text-align:center; font-size:26px; font-weight:bold; margin:16px 0; }
    .verdict-real { background: #1b6b3a; color:white; padding:18px; border-radius:8px; text-align:center; font-size:26px; font-weight:bold; margin:16px 0; }
    .signal-box { background:#f0f4f8; border-left: 4px solid #1a1a2e; padding:12px 16px; margin:8px 0; border-radius: 4px; }
    .disclaimer { border:1px solid #e9a800; background:#fff8e1; padding: 14px; border-radius: 6px; font-size:13px; margin-top:24px; }
</style>
''', unsafe_allow_html=True)

# 3. Load saved model and vectorizer using Joblib [cite: 571, 572, 575]
@st.cache_resource
def load_model():
    try:
        # Joblib use kar rahe hain taaki Python version mismatch na ho
        model = joblib.load('model.pkl')
        vec = joblib.load('vectorizer.pkl')
        return model, vec
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.info("Make sure 'model.pkl' and 'vectorizer.pkl' are in your GitHub repository.")
        return None, None

model, vectorizer = load_model()

# 4. Prediction function with signal explanation [cite: 579, 580, 581, 582, 586, 588, 596, 597]
def predict_with_explanation(headline):
    if model is None or vectorizer is None:
        return None, 0, []
    
    vec_input = vectorizer.transform([headline])
    prediction = model.predict(vec_input)[0]
    proba = model.predict_proba(vec_input)[0]
    confidence = max(proba) * 100
    
    feature_names = vectorizer.get_feature_names_out()
    dense = vec_input.toarray()[0]
    fake_idx = list(model.classes_).index('FAKE')
    
    # Machine learning ne jo weights seekhe hain unhe nikalna [cite: 592, 596]
    word_scores = dense * model.feature_log_prob_[fake_idx]
    top_indices = word_scores.argsort()[-3:][::-1]
    top_signals = [feature_names[i] for i in top_indices if dense[i] > 0]
    
    return prediction, confidence, top_signals

# 5. UI Layout [cite: 601, 602, 603, 604, 606, 613]
st.title('🔍 TruthLens')
st.subheader('AI-Powered Fake News Detector')
st.caption('Built from scratch over 8 weeks — by a Class 8 student')
st.divider()

headline_input = st.text_area(
    'Paste a news headline here:', 
    placeholder='e.g. SHOCKING: Miracle fruit destroys cancer overnight!',
    height=100
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyse = st.button('ANALYSE THIS HEADLINE', type='primary', use_container_width=True)

# 6. Results display logic [cite: 617, 618, 619, 620, 623, 627, 629, 631, 632]
if analyse:
    if len(headline_input.strip()) < 8:
        st.error('Please enter a headline of at least 8 characters.')
    else:
        with st.spinner('Analysing...'):
            pred, conf, signals = predict_with_explanation(headline_input)
            
            if pred == 'FAKE':
                st.markdown(f'<div class="verdict-fake">LIKELY FAKE NEWS ({conf:.0f}% Confidence)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-real">LIKELY REAL NEWS ({conf:.0f}% Confidence)</div>', unsafe_allow_html=True)
            
            if signals:
                st.write("**Top signals that influenced this verdict:**")
                for s in signals:
                    st.markdown(f'<div class="signal-box">The word or phrase <b>"{s}"</b> is strongly associated with {pred.lower()} news.</div>', unsafe_allow_html=True)

# 7. Responsible AI Disclaimer [cite: 634, 637, 638, 639, 640]
st.markdown('''
<div class="disclaimer">
    <strong>Important:</strong> This tool is an AI assistant, not a final judge. 
    It makes mistakes. Always verify surprising claims with trusted sources (BBC, NDTV, etc.) before sharing.
</div>
''', unsafe_allow_html=True)
