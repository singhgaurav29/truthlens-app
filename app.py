%%writefile app.py
import streamlit as st
import joblib
import time

# 1. Load the pre-trained machine learning model and vectorizer with caching
@st.cache_resource
def load_models():
    vec = joblib.load('vectorizer.pkl')
    mod = joblib.load('model.pkl')
    return vec, mod

# 2. Page Setup
st.set_page_config(page_title="TruthLens AI", page_icon="🕵️‍♂️", layout="centered")

# 3. Premium CSS Injection (Professional Clean Orange & White Theme)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Modern Gradient Title Styling */
        .main-title {
            font-size: 3.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #FF6A00 0%, #FF4500 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0px;
            padding-bottom: 0px;
            letter-spacing: -1px;
        }
        
        /* Subtitle Styling */
        .sub-title {
            text-align: center;
            font-size: 1.1rem;
            font-weight: 600;
            opacity: 0.7;
            letter-spacing: 1px;
            margin-top: -5px;
            margin-bottom: 35px;
        }
        
        /* High-End Interactive Orange Button */
        .stButton>button {
            background: linear-gradient(90deg, #FF6A00 0%, #FF4500 100%);
            color: #FFFFFF !important;
            border: none;
            border-radius: 8px;
            padding: 0.7rem;
            font-size: 1.1rem;
            font-weight: bold;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(255, 106, 0, 0.2);
        }
        
        .stButton>button:hover {
            box-shadow: 0 6px 18px rgba(255, 106, 0, 0.4);
            transform: translateY(-1px);
        }
        
        /* Probability metric label */
        .prob-text {
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 10px;
            margin-bottom: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

# Load Models Safely
try:
    vectorizer, model = load_models()
except Exception:
    st.error("⚠️ System Error: 'model.pkl' or 'vectorizer.pkl' is missing from the repository.")
    st.stop()

# 4. Interface Header (Original Context)
st.markdown('<h1 class="main-title">TRUTHLENS</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Fake News Detector | Built by Gaurav</p>', unsafe_allow_html=True)

# 5. User Input Section
headline = st.text_input(
    "Input Field", 
    placeholder="Enter a news headline to verify its authenticity...",
    label_visibility="collapsed"
)

st.write("") # Spacer

# 6. Prediction and Analysis Logic
if st.button("Analyze Headline", use_container_width=True):
    if headline.strip() == "":
        st.warning("⚠️ Please enter a headline to analyze!")
    else:
        # Spinner block
        with st.spinner("AI is analyzing patterns, context, and vocabulary..."):
            time.sleep(1.2)
            
            # Transforming text and executing model prediction
            text_vector = vectorizer.transform([headline])
            prediction = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]
            
            # Fetching probabilities based on original text framework
            fake_prob = probabilities[list(model.classes_).index('FAKE')]
            real_prob = probabilities[list(model.classes_).index('REAL')]
            
        st.markdown("### AI Analysis Result:")
        
        # 7. Dynamic Adaptive Output Display
        if prediction == 'FAKE':
            with st.container(border=True):
                st.error("🚨 **LIKELY FAKE NEWS**")
                
                # Probability Metric Reporting
                st.markdown(f"<div class='prob-text' style='color: #FF3B30;'>Fakeness Probability: {fake_prob*100:.1f}%</div>", unsafe_allow_html=True)
                st.progress(float(fake_prob))
                
                st.write("Our AI detected patterns commonly found in misinformation, clickbait, or fabricated news. Please verify with credible sources before sharing.")
        else:
            st.balloons()
            with st.container(border=True):
                st.success("✅ **LIKELY REAL NEWS**")
                
                # Probability Metric Reporting
                st.markdown(f"<div class='prob-text' style='color: #34C759;'>Authenticity Probability: {real_prob*100:.1f}%</div>", unsafe_allow_html=True)
                st.progress(float(real_prob))
                
                st.write("This headline appears credible and matches patterns of legitimate journalism. However, always stay critical!")%%writefile app.py
import streamlit as st
import joblib
import time

# 1. Load the pre-trained machine learning model and vectorizer with caching
@st.cache_resource
def load_models():
    vec = joblib.load('vectorizer.pkl')
    mod = joblib.load('model.pkl')
    return vec, mod

# 2. Page Setup
st.set_page_config(page_title="TruthLens AI", page_icon="🕵️‍♂️", layout="centered")

# 3. Premium CSS Injection (Professional Clean Orange & White Theme)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Modern Gradient Title Styling */
        .main-title {
            font-size: 3.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #FF6A00 0%, #FF4500 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0px;
            padding-bottom: 0px;
            letter-spacing: -1px;
        }
        
        /* Subtitle Styling */
        .sub-title {
            text-align: center;
            font-size: 1.1rem;
            font-weight: 600;
            opacity: 0.7;
            letter-spacing: 1px;
            margin-top: -5px;
            margin-bottom: 35px;
        }
        
        /* High-End Interactive Orange Button */
        .stButton>button {
            background: linear-gradient(90deg, #FF6A00 0%, #FF4500 100%);
            color: #FFFFFF !important;
            border: none;
            border-radius: 8px;
            padding: 0.7rem;
            font-size: 1.1rem;
            font-weight: bold;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(255, 106, 0, 0.2);
        }
        
        .stButton>button:hover {
            box-shadow: 0 6px 18px rgba(255, 106, 0, 0.4);
            transform: translateY(-1px);
        }
        
        /* Probability metric label */
        .prob-text {
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 10px;
            margin-bottom: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

# Load Models Safely
try:
    vectorizer, model = load_models()
except Exception:
    st.error("⚠️ System Error: 'model.pkl' or 'vectorizer.pkl' is missing from the repository.")
    st.stop()

# 4. Interface Header (Original Context)
st.markdown('<h1 class="main-title">TRUTHLENS</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Fake News Detector | Built by Gaurav</p>', unsafe_allow_html=True)

# 5. User Input Section
headline = st.text_input(
    "Input Field", 
    placeholder="Enter a news headline to verify its authenticity...",
    label_visibility="collapsed"
)

st.write("") # Spacer

# 6. Prediction and Analysis Logic
if st.button("Analyze Headline", use_container_width=True):
    if headline.strip() == "":
        st.warning("⚠️ Please enter a headline to analyze!")
    else:
        # Spinner block
        with st.spinner("AI is analyzing patterns, context, and vocabulary..."):
            time.sleep(1.2)
            
            # Transforming text and executing model prediction
            text_vector = vectorizer.transform([headline])
            prediction = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]
            
            # Fetching probabilities based on original text framework
            fake_prob = probabilities[list(model.classes_).index('FAKE')]
            real_prob = probabilities[list(model.classes_).index('REAL')]
            
        st.markdown("### AI Analysis Result:")
        
        # 7. Dynamic Adaptive Output Display
        if prediction == 'FAKE':
            with st.container(border=True):
                st.error("🚨 **LIKELY FAKE NEWS**")
                
                # Probability Metric Reporting
                st.markdown(f"<div class='prob-text' style='color: #FF3B30;'>Fakeness Probability: {fake_prob*100:.1f}%</div>", unsafe_allow_html=True)
                st.progress(float(fake_prob))
                
                st.write("Our AI detected patterns commonly found in misinformation, clickbait, or fabricated news. Please verify with credible sources before sharing.")
        else:
            st.balloons()
            with st.container(border=True):
                st.success("✅ **LIKELY REAL NEWS**")
                
                # Probability Metric Reporting
                st.markdown(f"<div class='prob-text' style='color: #34C759;'>Authenticity Probability: {real_prob*100:.1f}%</div>", unsafe_allow_html=True)
                st.progress(float(real_prob))
                
                st.write("This headline appears credible and matches patterns of legitimate journalism. However, always stay critical!")
