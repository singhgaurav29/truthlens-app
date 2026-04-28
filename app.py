import streamlit as st
import joblib

# 1. Apne saved model ko load karna
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

# 2. UI Design
st.set_page_config(page_title="TruthLens AI", page_icon="🕵️‍♂️")
st.title("TruthLens: Fake News Detector 🕵️‍♂️")
st.markdown("**Built by Gaurav** | Powered by Machine Learning")
st.write("---")

# 3. User Input
headline = st.text_input("Test karne ke liye koi bhi news headline type karein:", placeholder="E.g., Hillary Clinton caught in new video scandal...")

# 4. Button aur Logic
if st.button("Analyze Headline"):
    if headline.strip() == "":
        st.warning("⚠️ Bhai, pehle koi headline toh likho!")
    else:
        # User ke text ko numbers mein badalna
        text_vector = vectorizer.transform([headline])
        
        # Model se prediction lena
        prediction = model.predict(text_vector)[0]
        
        # Result dikhana
        st.write("### AI Analysis Result:")
        if prediction == 'FAKE':
            st.error("🚨 **LIKELY FAKE NEWS** - Be careful before sharing this!")
        else:
            st.success("✅ **LIKELY REAL NEWS** - Looks credible.")
