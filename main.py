import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
import pymupdf
from dotenv import load_dotenv  # Correct usage

# --- Load environment variables from .env ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API")

# --- Configure Gemini Model ---
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

# --- Helper Functions ---
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def analyze_resume(resume: str, jd: str) -> str:
    prompt = f"""You are a hiring assistant. Given a resume and a job description for 'Software Engineer', score the match from 0 to 10, and explain the decision.\nResume: {resume}\nJob Description: {jd}"""
    response = model.generate_content(prompt)
    return response.text

def analyze_feedback(feedback: str) -> str:
    prompt = f"""You are an HR analytics expert. Analyze the following feedback and predict:
1. Is the employee likely to leave soon? (Yes/No)
2. What are the key pain points?
3. Suggest specific strategies for HR to improve engagement.\nFeedback: {feedback}"""
    response = model.generate_content(prompt)
    return response.text

# --- Streamlit Interface ---
st.title("AI HR Assistant — Resume + Feedback Analysis")

tab1, tab2 = st.tabs(["Resume Screening", "Feedback Analysis"])

with tab1:
    st.header("Upload Resumes and Paste Job Description")
    jd_text = st.text_area("Paste Software Engineer Job Description")

    resume_files = st.file_uploader("Upload Resumes (.txt or .pdf)", type=["txt", "pdf"], accept_multiple_files=True)

    if st.button("Analyze Resumes") and resume_files and jd_text:
        for file in resume_files:
            if file.type == "application/pdf":
                resume_text = extract_text_from_pdf(file)
            else:
                resume_text = file.read().decode("utf-8")

            result = analyze_resume(resume_text, jd_text)
            st.subheader(f"Result for {file.name}")
            st.markdown(result)

with tab2:
    st.header("Upload Employee Feedback CSV")

    feedback_file = st.file_uploader("Upload CSV with 'feedback' column", type=["csv"])

    if st.button("Analyze Feedback") and feedback_file:
        df = pd.read_csv(feedback_file)
        if "feedback" not in df.columns:
            st.error("CSV must contain a 'feedback' column.")
        else:
            for i, row in df.iterrows():
                st.markdown(f"### Feedback {i+1}")
                result = analyze_feedback(row['feedback'])
                st.markdown(result)
