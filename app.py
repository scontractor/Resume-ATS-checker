from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
import io
from transformers import pipeline

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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume_file = request.files['resume']
        job_description = request.form['job_description']

        resume_text = extract_text_from_pdf(resume_file)
        ats_score = calculate_ats_score(resume_text, job_description)

        return jsonify({'ats_score': f"{ats_score:.2f}%"})

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)