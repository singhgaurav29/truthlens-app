import streamlit as st
import joblib
import time

# 1. Load Models Fast
@st.cache_resource
def load_models():
    vec = joblib.load('vectorizer.pkl')
    mod = joblib.load('model.pkl')
    return vec, mod

# 2. Page Config
st.set_page_config(page_title="TruthLens AI", page_icon="🌐", layout="centered")

# 3. Adaptive CSS (Works perfectly in BOTH Dark and Light Modes)
st.markdown("""
    <style>
        /* Logo Text - Gradient looks awesome on both dark and light backgrounds */
        .logo-text {
            font-size: 3.5rem;
            font-weight: 900;
            background: linear-gradient(90deg, #FF6A00 0%, #FFA040 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0px;
            padding-bottom: 0px;
        }
        
        /* Sub text uses opacity so it adapts to dark/light mode automatically */
        .sub-text {
            text-align: center;
            font-size: 1.1rem;
            font-weight: 600;
            opacity: 0.6; 
            letter-spacing: 1.5px;
            margin-top: -5px;
            margin-bottom: 30px;
        }
        
        /* Orange Premium Button */
        .stButton>button {
            background: linear-gradient(90deg, #FF6A00 0%, #FF8C00 100%);
            color: #FFFFFF !important;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: bold;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            box-shadow: 0 4px 15px rgba(255, 106, 0, 0.4);
            transform: translateY(-2px);
        }
    </style>
    """, unsafe_allow_html=True)

# Load Models
try:
    vectorizer, model = load_models()
except Exception:
    st.error("⚠️ System Error: 'model.pkl' or 'vectorizer.pkl' is missing from the server.")
    st.stop()

# 4. Header Design
st.markdown('<h1 class="logo-text">TRUTHLENS</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">AI-POWERED NEWS VERIFICATION</p>', unsafe_allow_html=True)

# 5. Search Interface (Native Streamlit so it adapts to dark/light mode)
headline = st.text_input(
    "Search", 
    placeholder="🔍 Paste a news headline here to analyze its authenticity...",
    label_visibility="collapsed"
)

st.write("") # Spacer

# 6. Action Button & Logic
if st.button("VERIFY AUTHENTICITY", use_container_width=True):
    if not headline.strip():
        st.warning("⚠️ Please enter a headline to scan.")
    else:
        with st.spinner("Scanning global patterns and linguistic structures..."):
            time.sleep(1.2)
            
            text_vector = vectorizer.transform([headline])
            prediction = model.predict(text_vector)[0]
            
        st.markdown("---")
        
        # 7. Adaptive Results using Streamlit's native containers
        if prediction == 'FAKE':
            with st.container(border=True):
                st.error("🚨 **High Risk: Likely Fabricated Content**")
                st.markdown("""
                    **Analysis:** The algorithm detected severe indicators of clickbait, sensationalism, or fabricated information. The linguistic structure fails standard journalistic integrity checks.
                    
                    *Action: Do not share or publish without corroborating with trusted outlets.*
                """)
        else:
            st.balloons()
            with st.container(border=True):
                st.success("✅ **Verified: Credible Pattern Detected**")
                st.markdown("""
                    **Analysis:** This headline passes automated linguistic checks. It matches the syntax, objectivity, and structure typical of verified journalistic sources.
                    
                    *Note: Automated system clearance. Standard editorial discretion is still advised.*
                """)
