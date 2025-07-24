import os
import fitz
import docx2txt
import faiss
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string
from sentence_transformers import SentenceTransformer
import openai

# ✅ 配置 DeepSeek API（新版 openai >= 1.0 接口）
client = openai.OpenAI(
    api_key="sk-18aa838cc6614a26afce435ea5f45668",
    base_url="https://api.deepseek.com/v1"
)
MODEL = "deepseek-chat"

# ✅ 初始化嵌入模型与岗位数据
embedder = SentenceTransformer("all-MiniLM-L6-v2")
data_path = r"D:\airesume\ecommerce_resume_requirements.csv"
df = pd.read_csv(data_path)
job_texts = df['Full_Description'].astype(str).tolist()
job_embeddings = embedder.encode(job_texts, convert_to_numpy=True)

# ✅ 构建 FAISS 向量索引
dimension = job_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(job_embeddings)

# ✅ 初始化 Flask 应用
app = Flask(__name__)

# ✅ 提取上传文件中的文本内容
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

# ✅ 构造 Prompt + 调用 DeepSeek 模型（新接口）
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
    return response.choices[0].message.content.strip()

# ✅ HTML 页面模板
HTML = """
<html>
  <head><title>AI Resume Screening</title></head>
  <body>
    <h2>📄 AI Resume Screening (E-commerce)</h2>
    <form action="/screen" method="post" enctype="multipart/form-data">
      <p>Upload resume (.pdf/.docx):</p>
      <input type="file" name="resume" required><br><br>
      <input type="submit" value="Submit">
    </form>
    {% if result %}
      <hr>
      <h3>🧠 Screening Result:</h3>
      <pre>{{ result }}</pre>
    {% endif %}
  </body>
</html>
"""

# ✅ 路由：主页
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)

# ✅ 路由：处理简历筛选逻辑
@app.route("/screen", methods=["POST"])
def screen():
    file = request.files['resume']
    resume_text = extract_text(file)
    query_vec = embedder.encode([resume_text])
    D, I = index.search(query_vec, k=1)
    matched_job = job_texts[I[0][0]]
    result = query_deepseek(matched_job, resume_text)
    return render_template_string(HTML, result=result)

# ✅ 启动应用
if __name__ == "__main__":
    app.run(debug=True)
