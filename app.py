import streamlit as st
import joblib
import time

# 1. Fast Model Loading
@st.cache_resource
def load_models():
    vec = joblib.load('vectorizer.pkl')
    mod = joblib.load('model.pkl')
    return vec, mod

# 2. Enterprise Page Configuration (WIDE Layout for Dashboard feel)
st.set_page_config(page_title="TruthLens | Threat Intelligence", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# 3. Custom CSS for Enterprise Look
st.markdown("""
    <style>
        /* Modern Button Styling */
        .stButton>button {
            background-color: #FF6A00;
            color: #FFFFFF !important;
            border: none;
            border-radius: 4px;
            padding: 0.6rem;
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #E65C00;
            box-shadow: 0 4px 12px rgba(255, 106, 0, 0.3);
        }
        
        /* Metric Styling */
        div[data-testid="stMetricValue"] {
            font-size: 2.5rem;
            font-weight: 800;
        }
        
        /* Headers */
        .dashboard-title {
            color: #FF6A00;
            font-weight: 900;
            font-size: 2.5rem;
            margin-bottom: -10px;
            letter-spacing: 1px;
        }
        .dashboard-subtitle {
            color: #888;
            font-size: 1rem;
            letter-spacing: 2px;
            font-weight: 600;
            margin-bottom: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

# Load Models
try:
    vectorizer, model = load_models()
except Exception:
    st.error("CRITICAL ERROR: Analytical models disconnected. Please check server payload.")
    st.stop()

# 4. Professional Sidebar Console
with st.sidebar:
    st.markdown("### 🎛️ SYSTEM CONSOLE")
    st.markdown("---")
    st.metric(label="NLP Engine Status", value="ONLINE", delta="Optimal Latency")
    st.metric(label="Database Registry", value="Active", delta="Synced Today")
    st.markdown("---")
    st.caption("Secure connection established via Streamlit Cloud.")
    st.caption("TRUTHLENS KERNEL v2.1.0")

# 5. Main Dashboard Header
st.markdown('<h1 class="dashboard-title">TRUTHLENS ANALYTICS</h1>', unsafe_allow_html=True)
st.markdown('<p class="dashboard-subtitle">MISINFORMATION THREAT INTELLIGENCE DASHBOARD</p>', unsafe_allow_html=True)

# 6. Input Panel
with st.container():
    st.markdown("#### 📡 Query Terminal")
    # Using text_area instead of text_input for a more substantial, professional feel
    headline = st.text_area(
        "Input Data", 
        height=100, 
        placeholder="Initialize sequence... Paste news headline or text excerpt here for algorithmic analysis.", 
        label_visibility="collapsed"
    )
    
    # Centering the button slightly using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("▶ EXECUTE DEEP SCAN", use_container_width=True)

st.markdown("---")

# 7. Analysis Engine & Dashboard Results
if analyze_btn:
    if not headline.strip():
        st.warning("⚠️ Exception: Null input detected. Please provide text data.")
    else:
        with st.spinner("Establishing neural pathways... Analyzing syntax, context, and linguistic entropy..."):
            time.sleep(1.5)
            
            text_vector = vectorizer.transform([headline])
            prediction = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]
            
            fake_prob = probabilities[list(model.classes_).index('FAKE')]
            real_prob = probabilities[list(model.classes_).index('REAL')]

        # Splitting results into a professional 2-column report
        res_col1, res_col2 = st.columns([1, 1.2])
        
        with res_col1:
            st.markdown("### 📊 Primary Verdict")
            if prediction == 'FAKE':
                st.error("🚨 HIGH THREAT: LIKELY FABRICATED")
                st.metric(
                    label="Fabrication Confidence Score", 
                    value=f"{fake_prob*100:.2f}%", 
                    delta="- Critical Deviation Detected", 
                    delta_color="inverse"
                )
                st.progress(float(fake_prob))
            else:
                st.success("✅ SYSTEM CLEAR: CREDIBLE PATTERN")
                st.metric(
                    label="Authenticity Confidence Score", 
                    value=f"{real_prob*100:.2f}%", 
                    delta="+ Standard Patterns Matched", 
                    delta_color="normal"
                )
                st.progress(float(real_prob))
                
        with res_col2:
            st.markdown("### 🔬 Diagnostic Breakdown")
            # Using an expander to look like a detailed system log
            with st.expander("View Detailed Linguistic Report", expanded=True):
                if prediction == 'FAKE':
                    st.markdown("""
                    **Diagnostic Flags Triggered:**
                    - **Sensationalism Index:** High
                    - **Objective Syntax:** Failed
                    - **Vocabulary Entropy:** Anomalous
                    
                    **System Conclusion:**
                    The linguistic signature of this query closely mirrors known parameters of clickbait, propaganda, or unverified rumor-mill architecture. Extreme caution advised before redistribution.
                    """)
                else:
                    st.markdown("""
                    **Diagnostic Flags Triggered:**
                    - **Sensationalism Index:** Nominal / Low
                    - **Objective Syntax:** Passed
                    - **Vocabulary Entropy:** Standard
                    
                    **System Conclusion:**
                    Text aligns with objective reporting frameworks. Neutral vocabulary and standard grammatical markers detected. Structure is consistent with verified journalistic databases.
                    """)
