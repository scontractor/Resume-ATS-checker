from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
import io
from transformers import pipeline
import re

app = Flask(__name__)

# Load AI model for text classification
classifier = pipeline("zero-shot-classification")


def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def calculate_ats_score(resume_text, job_description):
    # Use AI model to classify resume text against job description
    result = classifier(resume_text, candidate_labels=[job_description])
    return result['scores'][0] * 100  # Convert to percentage


def get_keyword_matches(resume_text, job_description):
    # Extract keywords from job description (simple method, can be improved)
    job_keywords = set(re.findall(r'\b\w+\b', job_description.lower()))
    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))

    matches = job_keywords.intersection(resume_words)
    missing = job_keywords - resume_words

    return list(matches), list(missing)


def generate_suggestions(missing_keywords):
    suggestions = []
    for keyword in missing_keywords[:5]:
        suggestions.append(f"Consider adding '{keyword}' to your resume if relevant to your experience.")
        suggestions.append(f"The job description mentions '{keyword}'. If you have experience with this, include it.")
        suggestions.append(f"'{keyword}' appears to be important for this role. Try to incorporate it if applicable.")
    return suggestions


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume_file = request.files['resume']
        job_description = request.form['job_description']

        resume_text = extract_text_from_pdf(resume_file)
        ats_score = calculate_ats_score(resume_text, job_description)

        matches, missing = get_keyword_matches(resume_text, job_description)
        suggestions = generate_suggestions(missing)

        return jsonify({
            'ats_score': f"{ats_score:.2f}%",
            'keyword_matches': matches[:10],  # Limit to top 10 matches
            'missing_keywords': missing[:10],  # Limit to top 10 missing
            'suggestions': suggestions
        })

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)