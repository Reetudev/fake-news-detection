import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page Configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# Title
st.title("📰 Fake News Detection System")

st.markdown("""
### Detect Fake News Using Machine Learning & NLP

Paste any news article below and the system will predict whether
the news is **Real** or **Fake**.
""")

# Input box
news = st.text_area(
    "Enter News Article:",
    height=250,
    placeholder="Paste a news article here..."
)

# Prediction button
if st.button("🔍 Check News"):

    if news.strip() == "":
        st.warning("⚠️ Please enter a news article.")
    else:

        # Convert text into numerical features
        news_vector = vectorizer.transform([news])

        # Predict
        prediction = model.predict(news_vector)

        # Confidence Score
        probabilities = model.predict_proba(news_vector)
        confidence = probabilities.max() * 100

        st.divider()

        if prediction[0] == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")

        st.info(f"📊 Confidence Score: {confidence:.2f}%")

# Footer
st.markdown("---")
st.markdown(
    "Developed by **Reetu H Jeeral** | Data Science & Machine Learning Intern"
)