import streamlit as st
import joblib
import time

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🕵️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# MODEL LOADING
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
        "⚠️ Error loading model.pkl or vectorizer.pkl.\n\n"
        "Make sure both files are in the same folder as app.py."
    )
    st.stop()

# ---------------------------
# THEME TOGGLE
# ---------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

col1, col2, col3 = st.columns([6, 1, 1])

with col3:
    if st.button("🌓"):
        if st.session_state.theme == "dark":
            st.session_state.theme = "light"
        else:
            st.session_state.theme = "dark"

theme = st.session_state.theme

# ---------------------------
# THEME COLORS
# ---------------------------
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
# PREMIUM CSS
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

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

header {{
    visibility: hidden;
}}

.block-container {{
    padding-top: 2rem;
    max-width: 850px;
}}

.main-title {{
    text-align: center;
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(90deg,#FF6A00,#FF4500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}}

.sub-title {{
    text-align: center;
    color: {subtitle};
    font-size: 1.05rem;
    margin-bottom: 40px;
}}

/* Target Streamlit's Native Bordered Container for the Hero Card */
div[data-testid="stVerticalBlockBorderedTest"] {{
    background: {card} !important;
    border: 1px solid {border} !important;
    border-radius: 24px !important;
    padding: 35px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15) !important;
}}

.stTextInput input {{
    background: {input_bg} !important;
    color: {text} !important;
    border: 2px solid {border} !important;
    border-radius: 14px !important;
    padding: 16px !important;
    font-size: 16px !important;
}}

.stTextInput input:focus {{
    border-color: #FF6A00 !important;
    box-shadow: 0 0 0 3px rgba(255,106,0,0.2) !important;
}}

.stButton > button {{
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 14px;
    font-size: 17px;
    font-weight: 700;
    color: white !important;
    background: linear-gradient(90deg,#FF6A00,#FF4500);
    transition: 0.3s;
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(255,106,0,0.35);
}}

.footer {{
    text-align: center;
    color: {subtitle};
    margin-top: 50px;
    font-size: 14px;
}}

</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# HEADER
# ---------------------------
st.markdown(
    """
<h1 class="main-title">TRUTHLENS AI</h1>
<p class="sub-title">
Fake News Detection Powered by Machine Learning
</p>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# INPUT CARD (Using native container syntax safely)
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
# ANALYSIS & PREDICTION
# ---------------------------
if analyze:

    if headline.strip() == "":
        st.warning("⚠️ Please enter a news headline to analyze.")
    else:

        # Professional Loading Animation
        with st.spinner("🧠 TruthLens AI is analyzing patterns and context..."):
            time.sleep(1.5)

            text_vector = vectorizer.transform([headline])
            prediction = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]
            classes = list(model.classes_)

            fake_prob = probabilities[classes.index("FAKE")]
            real_prob = probabilities[classes.index("REAL")]

        st.write("")
        st.markdown("## 📊 Analysis Result")

        # ---------------------------
        # FAKE NEWS RESULT
        # ---------------------------
        if prediction == "FAKE":

            st.markdown(
                f"""
                <div style="
                    background:#3B0A0A;
                    border:1px solid #EF4444;
                    border-radius:20px;
                    padding:25px;
                    margin-top:15px;
                ">
                    <h2 style="color:#F87171; margin-top:0;">
                        🚨 Likely Fake News
                    </h2>

                    <h1 style="
                        margin-bottom:0;
                        margin-top:10px;
                        color:#FFFFFF;
                        font-size: 3.5rem;
                        font-weight: 800;
                    ">
                        {fake_prob*100:.1f}%
                    </h1>

                    <p style="
                        color:#FCA5A5;
                        margin-top:0;
                    ">
                        Confidence Score
                    </p>

                    <hr style="
                        border:0;
                        border-top:1px solid rgba(255,255,255,0.15);
                        margin: 20px 0;
                    ">

                    <p style="
                        color:#F3F4F6;
                        line-height:1.7;
                        margin-bottom:0;
                    ">
                        TruthLens AI detected patterns commonly associated
                        with misinformation, sensationalism, clickbait,
                        or fabricated reporting.
                        <br><br>
                        Please verify this information using credible
                        news sources before sharing.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.write("")
            st.progress(float(fake_prob))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fake Probability", f"{fake_prob*100:.1f}%")
            with col2:
                st.metric("Real Probability", f"{real_prob*100:.1f}%")

        # ---------------------------
        # REAL NEWS RESULT
        # ---------------------------
        else:

            st.balloons()

            st.markdown(
                f"""
                <div style="
                    background:#052E16;
                    border:1px solid #22C55E;
                    border-radius:20px;
                    padding:25px;
                    margin-top:15px;
                ">
                    <h2 style="color:#4ADE80; margin-top:0;">
                        ✅ Likely Real News
                    </h2>

                    <h1 style="
                        margin-bottom:0;
                        margin-top:10px;
                        color:#FFFFFF;
                        font-size: 3.5rem;
                        font-weight: 800;
                    ">
                        {real_prob*100:.1f}%
                    </h1>

                    <p style="
                        color:#86EFAC;
                        margin-top:0;
                    ">
                        Confidence Score
                    </p>

                    <hr style="
                        border:0;
                        border-top:1px solid rgba(255,255,255,0.15);
                        margin: 20px 0;
                    ">

                    <p style="
                        color:#F3F4F6;
                        line-height:1.7;
                        margin-bottom:0;
                    ">
                        This headline aligns with patterns commonly found
                        in legitimate journalism and credible reporting.
                        <br><br>
                        Even when information appears trustworthy,
                        critical thinking and cross-checking sources
                        remain essential.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.write("")
            st.progress(float(real_prob))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Real Probability", f"{real_prob*100:.1f}%")
            with col2:
                st.metric("Fake Probability", f"{fake_prob*100:.1f}%")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown(
    """
    <div class="footer">
        🕵️ TruthLens AI • Powered by Machine Learning • Built by Gaurav
    </div>
    """,
    unsafe_allow_html=True
)
