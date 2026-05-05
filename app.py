import streamlit as st
import joblib
import time

# 1. Load Models Fast
@st.cache_resource
def load_models():
    vec = joblib.load('vectorizer.pkl')
    mod = joblib.load('model.pkl')
    return vec, mod

# 2. Modern Page Config
st.set_page_config(page_title="TruthLens", page_icon="🌐", layout="centered", initial_sidebar_state="collapsed")

# 3. Premium CSS Injection (Overrides Dark Mode elements gracefully)
st.markdown("""
    <style>
        /* Sleek Header styling */
        .header-container {
            text-align: center;
            padding: 2rem 0 1.5rem 0;
        }
        .logo-text {
            font-size: 3.5rem;
            font-weight: 900;
            background: linear-gradient(90deg, #FF6A00 0%, #FF3D00 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            line-height: 1.2;
            letter-spacing: -1px;
        }
        .sub-text {
            color: #888;
            font-size: 1.1rem;
            font-weight: 500;
            letter-spacing: 1px;
            margin-top: 0.5rem;
        }
        
        /* Modern Button Styling */
        .stButton>button {
            background: linear-gradient(90deg, #FF6A00 0%, #FF9100 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(255, 106, 0, 0.2);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 106, 0, 0.4);
        }
        
        /* Clean Result Cards */
        .result-card {
            padding: 1.5rem;
            border-radius: 12px;
            margin-top: 1.5rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            background-color: white;
            border: 1px solid #EAEAEA;
        }
        .fake-news { border-left: 6px solid #FF3B30; }
        .real-news { border-left: 6px solid #34C759; }
        
        /* Input label fix */
        .stTextInput label {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

# Load Models
try:
    vectorizer, model = load_models()
except Exception:
    st.error("System Error: model.pkl or vectorizer.pkl missing.")
    st.stop()

# 4. Clean Header
st.markdown("""
    <div class="header-container">
        <h1 class="logo-text">TRUTHLENS</h1>
        <p class="sub-text">AI-POWERED NEWS VERIFICATION</p>
    </div>
""", unsafe_allow_html=True)

# 5. Search Bar Interface
headline = st.text_input(
    "Search", 
    placeholder="🔍 Paste a news headline here to analyze its authenticity..."
)

st.write("") # Spacer

# 6. Action Button & Logic
if st.button("VERIFY AUTHENTICITY"):
    if not headline.strip():
        st.warning("⚠️ Please enter a headline to scan.")
    else:
        with st.spinner("Scanning global patterns and linguistic structures..."):
            time.sleep(1.2)
            text_vector = vectorizer.transform([headline])
            prediction = model.predict(text_vector)[0]
            
        if prediction == 'FAKE':
            st.markdown("""
                <div class="result-card fake-news">
                    <h2 style="color: #FF3B30; margin-top: 0; font-size: 1.6rem;">🚨 High Risk: Fabricated Content</h2>
                    <p style="color: #555; font-size: 1.05rem; line-height: 1.6;">
                        <strong>Analysis:</strong> The algorithm detected severe indicators of clickbait, sensationalism, or fabricated information. The linguistic structure fails standard journalistic integrity checks.
                    </p>
                    <p style="color: #888; font-size: 0.9rem; margin-bottom: 0;">
                        <em>Action: Do not share or publish without corroborating with trusted outlets.</em>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.balloons()
            st.markdown("""
                <div class="result-card real-news">
                    <h2 style="color: #34C759; margin-top: 0; font-size: 1.6rem;">✅ Verified: Credible Pattern</h2>
                    <p style="color: #555; font-size: 1.05rem; line-height: 1.6;">
                        <strong>Analysis:</strong> This headline passes automated linguistic checks. It matches the syntax, objectivity, and structure typical of verified journalistic sources.
                    </p>
                    <p style="color: #888; font-size: 0.9rem; margin-bottom: 0;">
                        <em>Note: Automated system clearance. Standard editorial discretion is still advised.</em>
                    </p>
                </div>
            """, unsafe_allow_html=True)
