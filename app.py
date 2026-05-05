import streamlit as st
import joblib
import time

# 1. Load the pre-trained machine learning model and vectorizer
# Using st.cache_resource so it loads only once and makes the app much faster
@st.cache_resource
def load_models():
    vec = joblib.load('vectorizer.pkl')
    mod = joblib.load('model.pkl')
    return vec, mod

# 2. UI Configuration and Page Setup
st.set_page_config(page_title="TruthLens AI", page_icon="🕵️‍♂️", layout="centered")

# Custom CSS for Animations and Premium Styling
st.markdown("""
    <style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .main-header {
        font-size: 3.5rem !important;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        animation: pulse 2s infinite;
        margin-bottom: 0px;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #888;
        margin-top: 5px;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Models safely
try:
    vectorizer, model = load_models()
except Exception as e:
    st.error("⚠️ Error loading models! Please ensure 'model.pkl' and 'vectorizer.pkl' exist in your repository.")
    st.stop()

# 3. Main Interface Design
st.markdown('<p class="main-header">TruthLens AI 🕵️‍♂️</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Fake News Detection System<br><b>Built by Gaurav</b></p>', unsafe_allow_html=True)

st.write("---")

# 4. User Input
headline = st.text_input(
    "📝 Enter a news headline to verify its authenticity:", 
    placeholder="E.g., SHOCKING: Doctors find a new miracle cure for all diseases..."
)

# Spacer
st.write("")

# 5. Button and Prediction Logic
if st.button("🔍 Analyze Truth", use_container_width=True):
    if headline.strip() == "":
        st.warning("⚠️ Please enter a headline to analyze!")
    else:
        # Animated loading spinner
        with st.spinner("🤖 AI is analyzing patterns, context, and vocabulary..."):
            time.sleep(1.5) # Simulating deep processing time for a cool effect
            
            # Convert text to numerical vectors
            text_vector = vectorizer.transform([headline])
            
            # Get prediction from the Naive Bayes model
            prediction = model.predict(text_vector)[0]
            
        # Display Results
        st.markdown("### 📊 AI Analysis Result:")
        
        if prediction == 'FAKE':
            st.error("🚨 **LIKELY FAKE NEWS**")
            st.warning("Our AI detected patterns commonly found in misinformation, clickbait, or fabricated news. Please verify with credible sources before sharing.")
        else:
            st.balloons() # Triggers Streamlit's built-in balloon animation
            st.success("✅ **LIKELY REAL NEWS**")
            st.info("This headline appears credible and matches patterns of legitimate journalism. However, always stay critical!")
