import streamlit as st
import joblib
import PyPDF2
import re
import openai
from dotenv import load_dotenv
import os
from openai import OpenAI
import plotly.graph_objects as go


load_dotenv()

# TO-DO - use env variable to hide the api key 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load Model
model = joblib.load("xgboost_resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

# Define job-specific expected skills
job_skills = {
    "BUSINESS-DEVELOPMENT": {
        "Sales Strategy", "Market Analysis", "Lead Generation", "B2B Sales", "B2C Sales",
        "Negotiation", "CRM Tools", "Strategic Partnerships", "Customer Retention",
        "Business Intelligence", "Go-to-Market Strategy", "Pitching", "Revenue Forecasting"
    },

    "ENGINEERING": {
        "Python", "C++", "C", "Java", "MATLAB", "Machine Learning", "TensorFlow", "PyTorch",
        "Computer Vision", "Embedded Systems", "Signal Processing", "Control Systems",
        "AutoCAD", "SolidWorks", "Robotics", "Circuit Design", "System Architecture",
        "Agile Methodologies", "CI/CD", "Version Control (Git)"
    },

    "FINANCE": {
        "Financial Analysis", "Financial Modeling", "Risk Management", "Investment Strategies",
        "Excel", "Power BI", "Data Analytics", "Budgeting", "Forecasting", "Accounting",
        "Corporate Finance", "Wealth Management", "Quantitative Analysis", "FinTech Tools",
        "Cryptocurrency Knowledge", "Regulatory Compliance"
    },

    "HEALTHCARE": {
        "Patient Care", "Medical Diagnosis", "Clinical Research", "First Aid", "EMR Systems",
        "Telemedicine", "Healthcare Analytics", "Phlebotomy", "Health Education",
        "Infection Control", "Medical Coding", "Basic Life Support (BLS)",
        "Electronic Health Records", "Nursing Skills", "Pharmacology"
    },

    "MARKETING": {
        "SEO", "SEM", "Google Ads", "Meta Ads", "Email Marketing", "Social Media Marketing",
        "Content Creation", "Content Strategy", "Influencer Marketing", "Marketing Automation",
        "Google Analytics (GA4)", "A/B Testing", "Copywriting", "Brand Management",
        "CRM Tools", "Adobe Creative Suite", "Video Marketing", "Campaign Management"
    }
}


# Skill categories for Radar Chart Visualization
skill_categories = {
    "Technical": {
        "Python", "Java", "C++", "C", "Go", "Rust", "JavaScript", "TypeScript", "SQL", "NoSQL",
        "Machine Learning", "Deep Learning", "Data Science", "Data Structures", "Algorithms",
        "Natural Language Processing", "Computer Vision", "Big Data", "Cloud Computing",
        "DevOps", "Cybersecurity", "Software Engineering", "Mobile Development", 
        "Web Development", "API Development", "Blockchain", "AR/VR Development", 
        "IoT", "Embedded Systems", "Robotics", "Quantum Computing", "Edge Computing"
    },

    "Tools": {
        "TensorFlow", "PyTorch", "Scikit-learn", "Keras", "XGBoost", "LightGBM",
        "Pandas", "NumPy", "Matplotlib", "Seaborn", "Tableau", "Power BI",
        "Excel", "Jupyter", "Google Colab", "VS Code", "Git", "GitHub", "GitLab",
        "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Firebase", "Heroku", 
        "Linux", "Bash", "Postman", "JIRA", "Figma", "Adobe XD", "Unity", "Unreal Engine",
        "Google Ads", "Meta Ads", "GA4", "Notion", "Slack", "Trello"
    },

    "Soft Skills": {
        "Communication", "Leadership", "Teamwork", "Negotiation", "Time Management", 
        "Adaptability", "Critical Thinking", "Problem Solving", "Creativity", 
        "Emotional Intelligence", "Conflict Resolution", "Public Speaking", 
        "Decision Making", "Presentation Skills", "Collaboration", "Work Ethic",
        "Self-Motivation", "Attention to Detail", "Project Management"
    }
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


def skill_match_score(found_skills, total_skills):
    if not total_skills:
        return 0

    return int((len(found_skills) / len(total_skills)) * 40) #max 40 points 




# list of powerful ats friendly keywords 
keywords = {"collaborated", "developed", "designed", "implemented", "analyzed", "optimized", "led"}


def keyword_score(resume_text):
    # makes sure that only whole words are fetched 
    # \b helps avoiding false-positives 
    found = [kw for kw in keywords if re.search(rf"\b{kw}\b", resume_text, re.IGNORECASE)]

    return int((len(found) / len(keywords)) * 30) #max 30 points 


def readability_score(resume_text):
    # split the resume text intpo sentences 
    sentences = re.split(r'[.!?]', resume_text)

    # remove empty strings and extra whitespaces 
    # s.strip() removes all the extra whitespaces 
    sentences = [s.strip() for s in sentences if s.strip()]

    # return neutral score if there are no valid sentences 
    if not sentences:
     return 10

    # Calculate average sentence length in number of words
    avg_length = sum(len(s.split()) for s in sentences) / len(sentences)

    # Return full 20 points if avg sentence length is between 12–20 words (considered optimal)
    # Else return 10 points for non-optimal length
    return 20 if 12 < avg_length < 20 else 10  #max score is 20 


def length_score(resume_text):
    # split resume into words using spaces 
    words = resume_text.split()

    # awards 10 points if the resume has 200 to 1000 words 
    # else 5 points 
    return 10 if 200 <= len(words) <= 1000 else 5

def get_resume_score(resume_text, found_skills, total_skills):
    # Combines 4 different components to compute a final score out of 100:
    return (
        skill_match_score(found_skills, total_skills) +  # Max 40
        keyword_score(resume_text) +                      # Max 30
        readability_score(resume_text) +                  # Max 20
        length_score(resume_text)                         # Max 10
    )

def get_skill_match_by_category(resume_text):
 
 """
 Check how many skills match in each category and return percentage match.

 Returns:
        A dictionary with category names as keys and percentage match as values.
        Example: {"Technical": 40, "Tools": 60, "Soft Skills": 25}
        
 """
 category_scores = {}

 # iterate through each category and its skill set 
 for category, skills in skill_categories.items():
  # use regex to find how many skills are actua;;y mentioned in the resume 
  # re.escape(skill) ensures special characters like C++ are treated safely 
  found = {skill for skill in skills if re.search(rf"\b{re.escape(skill)}\b", resume_text, re.IGNORECASE) }

  # calculate match %
  percentage = int((len(found) / len(skills)) * 100)

  # save the result in the dictionary 
  category_scores[category] = percentage

 return category_scores


def plot_radar_chart(skill_scores):
 """
    Creates a radar (spider) chart from skill category scores.
    This helps visually understand strengths across categories.
    
    Args:
        skill_scores: A dictionary like {"Technical": 40, "Tools": 60, "Soft Skills": 25}

    Returns:
        A Plotly radar chart figure.
    """
 # get categories and their corresponding values 
 categories = list(skill_scores.keys())
 values = list(skill_scores.values())


 # repeat the first value at the end to complete the circular shape 
 values += values[:1] #radar chart connects back to the first point 


 fig = go.Figure(
  data = [
   go.Scatterpolar(
    r = values,   #radius values (%)
    theta = categories + [categories[0]],  #angles (axes)
    fill = 'toself',             # fill inside the shape
    name = 'Skill Match'
     
   )
  ],
  layout = go.Layout(
   polar = dict(
    radialaxis = dict(
     visible = True,     #show cicular axis
     range = [0,100]    #range from 0 to 100%
    )
   ),
   showlegend = True
  )
 )

 return fig
 


# ----------------- AI Feedback Helper Functions -----------------

# Function to create a prompt for the AI model to analyze the resume
def generate_feedback_prompt(resume_text):
    return f"""
You are an expert resume reviewer.

Review the following resume content and provide actionable feedback on:

1. Wording Improvements: Suggest stronger verbs or clearer phrases.
2. Grammar and Spelling Issues.
3. ATS Optimization: Are there keywords missing? Any passive voice?
4. Formatting suggestions if necessary.
5. Any missing sections (e.g., Projects, Experience, Education).

Respond in bullet points.

Resume:
\"\"\"
{resume_text}
\"\"\"
"""

# Function to call OpenAI API and return resume feedback
def get_ai_feedback(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content



# Function to get the job fit feedback 
def generate_job_fit_prompt(resume_text, job_description):
    return f"""
You are an expert hiring manager.

Given the candidate's resume and the job description, analyze the match.

Please provide a concise summary with:
1. How well the resume fits the job role
2. Strengths of the candidate
3. Areas where the resume is lacking
4. Suggestions for improvement if needed

Respond in a professional tone and bullet points.

Resume:
\"\"\"
{resume_text}
\"\"\"

Job Description:
\"\"\"
{job_description}
\"\"\"
"""


# Streamlit UI
st.title("📄 Resume Analyzer Dashboard")
st.write("Upload a resume in **PDF format**, and we will predict its job category and highlight missing skills!")

# File uploader
uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])

# Add the job description here 
st.subheader("📄 Paste the Job Description")
job_description = st.text_area("Paste the job description here", height=200)


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


    score = get_resume_score(resume_text, found_skills, job_skills.get(predicted_category, set()))

    st.subheader("📊 Resume Score")
    st.info(f"Your resume score **{score}/100** based on skill match, keywords, readability, and length" )

    # ADDING THE PIE CHART IMPLEMENTATION HERE 
    st.subheader("📈 Skill Match by Category")

    skill_scores = get_skill_match_by_category(resume_text)

    fig = plot_radar_chart(skill_scores)

    st.plotly_chart(fig, use_container_width = True)

    # AI Powered feedback button 
    if st.button("💡 Get AI-Powered Feedback"):
     with st.spinner("Analysing your resume..."):
      
      prompt = generate_feedback_prompt(resume_text)
      feedback = get_ai_feedback(prompt)
      st.subheader("🧠 AI-Powered Resume Suggestions")
      st.markdown(feedback)


    if job_description and st.button("🧠 Analyze Job Fit"):
     with st.spinner("Evaluating match with job description..."):
         job_fit_prompt = generate_job_fit_prompt(resume_text, job_description)
         job_fit_summary = get_ai_feedback(job_fit_prompt)
         st.subheader("🔍 AI-Powered Job Fit Summary")
         st.markdown(job_fit_summary)

      
