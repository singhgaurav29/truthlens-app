import streamlit as st
import joblib
import time

# ---------------------------
# 1. PAGE CONFIGURATION
# ---------------------------
st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🕵️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# 2. MODEL LOADING (WITH CACHING)
# ---------------------------
@st.cache_resource
def load_models():
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")
    return vectorizer, model

try:
    vectorizer, model = load_models()
except Exception:
    st.error(
        "⚠️ System Error: 'model.pkl' or 'vectorizer.pkl' missing.\n\n"
        "Please ensure both files are uploaded in the same repository folder as app.py."
    )
    st.stop()

# ---------------------------
# 3. INTERACTIVE THEME TOGGLE
# ---------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Alignment for theme toggle button
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    if st.button("🌓"):
        if st.session_state.theme == "dark":
            st.session_state.theme = "light"
        else:
            st.session_state.theme = "dark"

theme = st.session_state.theme

# Dynamic Theme Colors Configuration
if theme == "dark":
    bg = "#0F172A"
    card = "#1E293B"
    text = "#FFFFFF"
    subtitle = "#CBD5E1"
    input_bg = "#334155"
    border = "#475569"
else:
    bg = "#F8FAFC"
    card = "#FFFFFF"
    text = "#0F172A"
    subtitle = "#64748B"
    input_bg = "#FFFFFF"
    border = "#E2E8F0"

# ---------------------------
# 4. PREMIUM UI DESIGN (CSS INJECTION)
# ---------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {bg};
    color: {text};
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

.block-container {{
    padding-top: 1rem;
    max-width: 850px;
}}

.main-title {{
    text-align: center;
    font-size: 3.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FF6A00, #FF4500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    letter-spacing: -1px;
}}

.sub-title {{
    text-align: center;
    color: {subtitle};
    font-size: 1.1rem;
    font-weight: 500;
    margin-top: 5px;
    margin-bottom: 40px;
    opacity: 0.8;
}}

/* Custom Styling for Streamlit Native Container */
div[data-testid="stVerticalBlockBorderedTest"] {{
    background: {card} !important;
    border: 1px solid {border} !important;
    border-radius: 20px !important;
    padding: 30px !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1) !important;
}}

.stTextInput input {{
    background: {input_bg} !important;
    color: {text} !important;
    border: 2px solid {border} !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-size: 16px !important;
}}

.stTextInput input:focus {{
    border-color: #FF6A00 !important;
    box-shadow: 0 0 0 3px rgba(255,106,0,0.2) !important;
}}

.stButton > button {{
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
    font-weight: 700;
    color: white !important;
    background: linear-gradient(90deg, #FF6A00, #FF4500);
    transition: 0.3s ease;
    box-shadow: 0 4px 12px rgba(255, 106, 0, 0.15);
}}

.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(255,106,0,0.3);
}}

.footer {{
    text-align: center;
    color: {subtitle};
    margin-top: 60px;
    font-size: 14px;
    opacity: 0.7;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# 5. APPLICATION HEADER
# ---------------------------
st.markdown('<h1 class="main-title">TRUTHLENS AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Fake News Detection Powered by Machine Learning</p>', unsafe_allow_html=True)

# ---------------------------
# 6. CENTRAL INTERFACE CARD
# ---------------------------
with st.container(border=True):
    headline = st.text_input(
        "News Headline Input",
        placeholder="Paste a news headline here to verify authenticity...",
        label_visibility="collapsed"
    )

    analyze = st.button(
        "🔍 Analyze Headline",
        use_container_width=True
    )

# ---------------------------
# 7. ANALYTICS & PREDICTION LOGIC
# ---------------------------
if analyze:
    if headline.strip() == "":
        st.warning("⚠️ Please enter a news headline to analyze.")
    else:
        # Visual Processing Buffer
        with st.spinner("🧠 TruthLens AI is analyzing patterns and context..."):
            time.sleep(1.2)

            # Mathematical Transformation & Calculation
            text_vector = vectorizer.transform([headline])
            prediction = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]
            classes = list(model.classes_)

            fake_prob = probabilities[classes.index("FAKE")]
            real_prob = probabilities[classes.index("REAL")]

        st.write("")
        st.markdown("## 📊 Analysis Result")

        # --- CASE A: FAKE NEWS DETECTION ---
        if prediction == "FAKE":
            with st.container(border=True):
                st.markdown("### 🚨 Likely Fake News")
                st.markdown(f"# <span style='color:#F87171; font-weight:800;'>{fake_prob*100:.1f}%</span>", unsafe_allow_html=True)
                st.markdown("<p style='opacity:0.6; margin-top:-15px; font-weight:500;'>Confidence Score</p>", unsafe_allow_html=True)
                st.markdown("---")
                st.write(
                    "TruthLens AI detected structural patterns, syntax indicators, or contextual markers "
                    "commonly associated with misinformation, clickbait, or unverified reports."
                )
                st.write("Please cross-check this information using credible journalistic outlets before sharing.")
            
            st.write("")
            st.progress(float(fake_prob))

            # Metric Data Matrix
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Fake Probability", f"{fake_prob*100:.1f}%")
            with col_m2:
                st.metric("Real Probability", f"{real_prob*100:.1f}%")

        # --- CASE B: REAL NEWS DETECTION ---
        else:
            st.balloons()
            with st.container(border=True):
                st.markdown("### ✅ Likely Real News")
                st.markdown(f"# <span style='color:#4ADE80; font-weight:800;'>{real_prob*100:.1f}%</span>", unsafe_allow_html=True)
                st.markdown("<p style='opacity:0.6; margin-top:-15px; font-weight:500;'>Confidence Score</p>", unsafe_allow_html=True)
                st.markdown("---")
                st.write(
                    "This headline structural format aligns successfully with parameters commonly found "
                    "in legitimate journalism and verified objective reporting frameworks."
                )
                st.write("Even when a syntax check passes, standard factual verification across multiple sources is recommended.")
            
            st.write("")
            st.progress(float(real_prob))

            # Metric Data Matrix
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Real Probability", f"{real_prob*100:.1f}%")
            with col_m2:
                st.metric("Fake Probability", f"{fake_prob*100:.1f}%")

# ---------------------------
# 8. SYSTEM FOOTER
# ---------------------------
st.markdown(
    """
    <div class="footer">
        🕵️ TruthLens AI • Powered by Machine Learning 
    </div>
    """,
    unsafe_allow_html=True
)

