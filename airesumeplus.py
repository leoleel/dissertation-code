import os
import re
import fitz
import docx2txt
import faiss
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string
from sentence_transformers import SentenceTransformer
import openai

client = openai.OpenAI(
    api_key="sk-18aa838cc6614a26afce435ea5f45668",
    base_url="https://api.deepseek.com/v1"
)
MODEL = "deepseek-chat"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
data_path = r"D:\airesume\ecommerce_resume_requirements.csv"
df = pd.read_csv(data_path)
job_texts = df['Full_Description'].astype(str).tolist()
job_embeddings = embedder.encode(job_texts, convert_to_numpy=True)

index = faiss.IndexFlatL2(job_embeddings.shape[1])
index.add(job_embeddings)

app = Flask(__name__)

# 提取文本内容
def extract_text(file):
    if file.filename.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif file.filename.endswith(".docx"):
        path = "temp.docx"
        file.save(path)
        text = docx2txt.process(path)
        os.remove(path)
        return text
    return ""

# 提取分数
def extract_score(text):
    match = re.search(r"Match Score.*?(\d+(\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return -1.0

# 调用 deepseek 模型
def query_deepseek(job_text, resume_text):
    prompt = f"""You're an HR expert. Based on the job description below and the resume provided, answer:

Job Description:
{job_text}

Resume:
{resume_text[:3000]}

Please answer in this format:
1. Match (Yes/No)?
2. Strengths:
3. Weaknesses:
4. Match Score (0-10):"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    result = response.choices[0].message.content.strip()
    result = result.replace("1. Match", "<b>1. Match</b>")
    result = result.replace("2. Strengths", "<b>2. Strengths</b>")
    result = result.replace("3. Weaknesses", "<b>3. Weaknesses</b>")
    result = result.replace("4. Match Score", "<b>4. Match Score</b>")
    return result

HTML = """
<html>
<head>
  <title>AI Resume Screening</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f4f6f8; color: #333; }
    h2 { color: #2c3e50; }
    input[type=file], input[type=submit] { font-size: 16px; padding: 10px; margin-top: 10px; }
    form { background: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .result { margin-top: 30px; background: #ecf0f1; padding: 15px; border-left: 5px solid #2980b9; border-radius: 5px; }
    .resume-title { font-weight: bold; color: #2980b9; margin-top: 20px; }
    pre { white-space: pre-wrap; }
    .highlight { background: #dff0d8; border-left: 5px solid #27ae60; }
  </style>
</head>
<body>
  <h2>📄 AI Resume Screening (Ranked + Comparison)</h2>
  <form action="/screen" method="post" enctype="multipart/form-data">
    <p>Select one or more resumes (.pdf/.docx):</p>
    <input type="file" name="resumes" multiple required><br>
    <input type="submit" value="Submit All">
  </form>

  {% if results %}
    <hr>
    <h3>🏆 Top Ranked Resume Highlighted</h3>
    {% for name, result, top in results %}
      <div class="result {% if top %}highlight{% endif %}">
        <div class="resume-title">📌 {{ name }}</div>
        <pre>{{ result|safe }}</pre>
      </div>
    {% endfor %}
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)

@app.route("/screen", methods=["POST"])
def screen():
    files = request.files.getlist("resumes")
    scored = []
    for file in files:
        resume_text = extract_text(file)
        query_vec = embedder.encode([resume_text])
        D, I = index.search(query_vec, k=1)
        matched_job = job_texts[I[0][0]]
        result = query_deepseek(matched_job, resume_text)
        score = extract_score(result)
        scored.append((file.filename, result, score))

    # 排序 & 标记第一名
    scored.sort(key=lambda x: x[2], reverse=True)
    results = [(name, res, i == 0) for i, (name, res, _) in enumerate(scored)]

    return render_template_string(HTML, results=results)

if __name__ == "__main__":
    app.run(debug=True)
