import streamlit as st
import joblib
import PyPDF2
import re

# Load Model
model = joblib.load("xgboost_resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

# Define job-specific expected skills
job_skills = {
    "BUSINESS-DEVELOPMENT": {"Sales Strategy", "Market Analysis", "Lead Generation", "Negotiation"},
    "ENGINEERING": {"Python", "C++", "MATLAB", "Machine Learning", "TensorFlow"},
    "FINANCE": {"Financial Analysis", "Risk Management", "Investment Strategies", "Excel"},
    "HEALTHCARE": {"Patient Care", "Medical Diagnosis", "Clinical Research", "First Aid"},
    "MARKETING": {"SEO", "Google Ads", "Social Media Marketing", "Content Creation"},
}

# Function to extract text from PDF resumes
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text.strip()

# Function to compare extracted resume text with job-specific skills
def extract_resume_skills(resume_text, category):
    if category not in job_skills:
        return None, None  # No predefined skills for this category
    
    expected_skills = job_skills[category]
    found_skills = {skill for skill in expected_skills if re.search(rf"\b{skill}\b", resume_text, re.IGNORECASE)}
    missing_skills = expected_skills - found_skills  # Identify missing skills
    
    return found_skills, missing_skills

# Streamlit UI
st.title("📄 Resume Analyzer Dashboard")
st.write("Upload a resume in **PDF format**, and we will predict its job category and highlight missing skills!")

# File uploader
uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    resume_text = extract_text_from_pdf(uploaded_file)

    # Convert text to TF-IDF features
    text_vectorized = vectorizer.transform([resume_text])

    # Predict job category
    prediction = model.predict(text_vectorized)
    predicted_category = encoder.inverse_transform(prediction)[0]

    # Display prediction result
    st.success(f"📝 **Predicted Job Category:** {predicted_category}")

    # Extract missing skills
    found_skills, missing_skills = extract_resume_skills(resume_text, predicted_category)

    # Display missing skills feedback
    if missing_skills:
        st.warning("⚠️ **Missing Skills:**")
        st.write(", ".join(missing_skills))
    else:
        st.success("✅ **Your resume contains all the necessary skills for this category!**")

    # Show extracted text (optional)
    with st.expander("🔍 View Extracted Resume Text"):
        st.write(resume_text)
