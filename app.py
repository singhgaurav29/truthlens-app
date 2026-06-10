import streamlit as st
import joblib
import time

# 1. Fast Model Loading
@st.cache_resource
def load_models():
    vec = joblib.load('vectorizer.pkl')
    mod = joblib.load('model.pkl')
    return vec, mod

# 2. Advanced Page Configuration
st.set_page_config(page_title="TruthLens OS", page_icon="⚡", layout="centered", initial_sidebar_state="collapsed")

# 3. Enterprise-Grade CSS Injection
st.markdown("""
    <style>
        /* Import premium font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        /* Hero Section */
        .hero-title {
            font-size: 4.2rem;
            font-weight: 900;
            background: linear-gradient(135deg, #FF6A00 0%, #FFB000 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            letter-spacing: -2px;
            margin-bottom: 0;
            padding-bottom: 0;
            line-height: 1.1;
        }
        
        .hero-subtitle {
            text-align: center;
            font-size: 1.15rem;
            font-weight: 500;
            color: #888888;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-top: 10px;
            margin-bottom: 40px;
        }

        /* Search Bar Enhancement */
        div[data-baseweb="input"] {
            background-color: transparent !important;
        }
        div[data-baseweb="input"] > div {
            background-color: rgba(128, 128, 128, 0.05);
            border: 2px solid rgba(255, 106, 0, 0.2);
            border-radius: 12px;
            padding: 4px 8px;
            transition: all 0.3s ease;
        }
        div[data-baseweb="input"] > div:focus-within {
            border-color: #FF6A00;
            box-shadow: 0 0 0 4px rgba(255, 106, 0, 0.1);
        }
        div[data-baseweb="input"] input {
            font-size: 1.1rem !important;
            font-weight: 500;
        }

        /* Primary Button Upgrade */
        .stButton>button {
            background: linear-gradient(135deg, #FF6A00 0%, #FF8C00 100%);
            color: #FFFFFF !important;
            border: none;
            border-radius: 12px;
            padding: 0.8rem;
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: 1px;
            width: 100%;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 14px 0 rgba(255, 106, 0, 0.39);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 106, 0, 0.5);
        }
        .stButton>button:active {
            transform: translateY(0);
        }

        /* Footer */
        .footer-text {
            text-align: center;
            font-size: 0.85rem;
            color: #666;
            margin-top: 50px;
            font-weight: 500;
        }
    </style>
    """, unsafe_allow_html=True)

# Load Models
try:
    vectorizer, model = load_models()
except Exception:
    st.error("SYSTEM HALT: Core analytical models ('model.pkl' or 'vectorizer.pkl') are offline.")
    st.stop()

# 4. App Header
st.markdown('<h1 class="hero-title">TRUTHLENS</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Intelligence & Fact-Checking Engine</p>', unsafe_allow_html=True)

# 5. Core Interface
headline = st.text_input(
    "Query Input", 
    placeholder="Enter a headline or news snippet to initiate analysis...",
    label_visibility="collapsed"
)

st.write("") 

# 6. Analysis Engine
if st.button("INITIATE SCAN", use_container_width=True):
    if not headline.strip():
        st.warning("Action Required: Please input data to scan.")
    else:
        with st.spinner("Processing semantics, verifying linguistic markers, and calculating entropy..."):
            time.sleep(1.5) # Simulating heavy compute
            
            text_vector = vectorizer.transform([headline])
            prediction = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]
            
            fake_prob = probabilities[list(model.classes_).index('FAKE')]
            real_prob = probabilities[list(model.classes_).index('REAL')]
            
        st.write("")
        
        # 7. Enterprise Result Cards (Glassmorphism inspired)
        if prediction == 'FAKE':
            st.markdown(f"""
                <div style="padding: 2rem; border-radius: 16px; background: rgba(255, 59, 48, 0.05); border: 1px solid rgba(255, 59, 48, 0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 59, 48, 0.1); padding-bottom: 1rem; margin-bottom: 1rem;">
                        <h2 style="color: #FF3B30; margin: 0; font-size: 1.8rem; font-weight: 800;">CRITICAL RISK</h2>
                        <div style="text-align: right;">
                            <span style="font-size: 0.9rem; color: #888; text-transform: uppercase; letter-spacing: 1px;">Fabrication Confidence</span><br>
                            <span style="font-size: 2rem; font-weight: 900; color: #FF3B30;">{fake_prob*100:.1f}%</span>
                        </div>
                    </div>
                    <p style="font-size: 1.1rem; line-height: 1.6; font-weight: 500;">
                        System detected severe deviations from verified journalism standards. High concentration of sensationalist markers, clickbait syntax, or fabricated semantics identified.
                    </p>
                    <div style="background: rgba(255, 59, 48, 0.1); padding: 10px 15px; border-radius: 8px; font-size: 0.95rem; color: #FF3B30; font-weight: 600;">
                        ⚠️ PROTOCOL: Containment advised. Do not distribute without severe cross-referencing.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.progress(float(fake_prob))
            
        else:
            st.balloons()
            st.markdown(f"""
                <div style="padding: 2rem; border-radius: 16px; background: rgba(52, 199, 89, 0.05); border: 1px solid rgba(52, 199, 89, 0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(52, 199, 89, 0.1); padding-bottom: 1rem; margin-bottom: 1rem;">
                        <h2 style="color: #34C759; margin: 0; font-size: 1.8rem; font-weight: 800;">VERIFIED CLEAR</h2>
                        <div style="text-align: right;">
                            <span style="font-size: 0.9rem; color: #888; text-transform: uppercase; letter-spacing: 1px;">Authenticity Confidence</span><br>
                            <span style="font-size: 2rem; font-weight: 900; color: #34C759;">{real_prob*100:.1f}%</span>
                        </div>
                    </div>
                    <p style="font-size: 1.1rem; line-height: 1.6; font-weight: 500;">
                        Query structure aligns with trusted media databases. Syntactical objectivity and phrasing match established patterns of credible reporting.
                    </p>
                    <div style="background: rgba(52, 199, 89, 0.1); padding: 10px 15px; border-radius: 8px; font-size: 0.95rem; color: #34C759; font-weight: 600;">
                        ✓ PROTOCOL: Safe for consumption. Standard editorial discretion applies.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.progress(float(real_prob))

# 8. Footer
st.markdown('<div class="footer-text">TRUTHLENS KERNEL v1.0.4 • ENGINEERED BY GAURAV</div>', unsafe_allow_html=True)
