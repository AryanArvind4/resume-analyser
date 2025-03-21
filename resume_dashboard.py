import streamlit as st
import joblib
import PyPDF2

# Load Model
model = joblib.load("xgboost_resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("📄 Resume Analyzer Dashboard")
st.write("Upload a resume and get job category prediction!")

# File uploader
uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])

if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    predicted_category = encoder.inverse_transform(prediction)[0]

    st.success(f"📝 **Predicted Job Category:** {predicted_category}")
