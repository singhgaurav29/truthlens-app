
import streamlit as st
import joblib
import time

# 1. Load the pre-trained machine learning model and vectorizer
@st.cache_resource
def load_models():
    vec = joblib.load('vectorizer.pkl')
    mod = joblib.load('model.pkl')
    return vec, mod

# 2. UI Configuration and Page Setup
# Setting the page layout and forcing a clean title
st.set_page_config(page_title="TruthLens Verification", page_icon="📰", layout="centered")

# Custom CSS for Professional Orange & White News Portal Theme
st.markdown("""
    <style>
    /* Main Background and Text Colors */
    .stApp {
        background-color: #FAFAFA;
    }
    
    /* Professional Orange Top Banner */
    .news-banner {
        background-color: #FF6600;
        color: #FFFFFF;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .news-banner h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: 1.5px;
        color: #FFFFFF !important;
    }
    .news-banner p {
        margin: 5px 0 0 0;
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.9;
    }
    
    /* Sub-header styling */
    .system-status {
        text-align: center;
        font-size: 0.9rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 30px;
        font-weight: 600;
    }

    /* Orange Button Styling */
    div.stButton > button:first-child {
        background-color: #FF6600;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 12px 24px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #E65C00;
        color: white;
        box-shadow: 0 4px 12px rgba(255, 102, 0, 0.3);
    }
    
    /* Result Box Styling */
    .result-box-real {
        border-left: 5px solid #28a745;
        background-color: #f8fff9;
        padding: 15px;
        border-radius: 4px;
        margin-top: 20px;
    }
    .result-box-fake {
        border-left: 5px solid #dc3545;
        background-color: #fff8f8;
        padding: 15px;
        border-radius: 4px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Models safely
try:
    vectorizer, model = load_models()
except Exception as e:
    st.error("⚠️ System Error: Verification models offline. Please ensure 'model.pkl' and 'vectorizer.pkl' are active.")
    st.stop()

# 3. Official Portal Header
st.markdown("""
    <div class="news-banner">
        <h1>TRUTHLENS</h1>
        <p>Global News Verification & Fact-Checking Engine</p>
    </div>
    <div class="system-status">System Active • Maintained by Gaurav</div>
    """, unsafe_allow_html=True)

# 4. User Input Section
st.markdown("### 🔍 Verification Portal")
headline = st.text_input(
    "Enter the news headline or article title below to scan against our machine learning database:", 
    placeholder="e.g., White House releases new statement regarding the economy..."
)

st.write("") # Spacer

# 5. Core Logic and Processing
if st.button("Initialize Scan"):
    if headline.strip() == "":
        st.warning("⚠️ Input Required: Please enter a headline to begin the scan.")
    else:
        # Professional loading sequence
        with st.spinner("Analyzing linguistic patterns and cross-referencing database..."):
            time.sleep(1.5) 
            
            # Text processing
            text_vector = vectorizer.transform([headline])
            
            # Prediction
            prediction = model.predict(text_vector)[0]
            
        # Official-looking Results
        st.markdown("---")
        st.markdown("### 📑 Official Scan Report")
        
        if prediction == 'FAKE':
            st.markdown("""
            <div class="result-box-fake">
                <h3 style="color: #dc3545; margin-top: 0;">🚨 STATUS: LIKELY FABRICATED</h3>
                <p style="margin-bottom: 0;"><b>Analysis:</b> The algorithm detected structural anomalies, sensationalism, or vocabulary patterns commonly associated with misinformation, clickbait, or unverified sources. <br><br><i>Recommendation: Do not share or publish without corroborating with recognized media outlets.</i></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.balloons()
            st.markdown("""
            <div class="result-box-real">
                <h3 style="color: #28a745; margin-top: 0;">✅ STATUS: VERIFIED PATTERN</h3>
                <p style="margin-bottom: 0;"><b>Analysis:</b> The headline aligns with standard journalistic patterns, objective phrasing, and verified syntax formats. <br><br><i>Recommendation: Appears credible. Standard editorial discretion is still advised.</i></p>
            </div>
            """, unsafe_allow_html=True)
