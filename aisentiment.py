# # prepare_embeddings.py
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
#
# df = pd.read_csv(r"D:\airesume\amazon_reviews.csv")
# df = df.dropna(subset=["verified_reviews", "feedback"])
# texts = df["verified_reviews"].astype(str).tolist()
#
# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = embedder.encode(texts, convert_to_numpy=True)
#
# np.save(r"D:\airesume\sentiment_embeddings.npy", embeddings)
# df[["verified_reviews", "feedback"]].to_csv(r"D:\airesume\sentiment_texts_labels.csv", index=False)
#
# print("✅ 训练数据预处理完成")
import os
import numpy as np
import pandas as pd
import faiss
from flask import Flask, request, render_template_string
from sentence_transformers import SentenceTransformer
import openai

client = openai.OpenAI(
    api_key="sk-18aa838cc6614a26afce435ea5f45668",  # 替换为你自己的 Key
    base_url="https://api.deepseek.com/v1"
)
MODEL = "deepseek-chat"

# 预加载训练数据嵌入与语料
embed_path = r"D:\airesume\sentiment_embeddings.npy"
text_path = r"D:\airesume\sentiment_texts_labels.csv"
embeddings = np.load(embed_path)
df = pd.read_csv(text_path)
texts = df["verified_reviews"].tolist()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

app = Flask(__name__)

HTML = """
<html>
<head>
  <title>AI Sentiment Analyzer</title>
  <style>
    body { font-family: Arial; background: #f9f9fb; padding: 30px; color: #333; }
    h2 { color: #2c3e50; }
    form, .result { background: #fff; padding: 20px; border-radius: 10px; margin-top: 20px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    textarea, input[type=file], input[type=submit] { font-size: 16px; margin-top: 10px; width: 100%; padding: 10px; }
    .result { white-space: pre-wrap; border-left: 5px solid #2980b9; }
  </style>
</head>
<body>
  <h2>🧠 AI Sentiment Analyzer</h2>
  <form method="POST" enctype="multipart/form-data">
    <label>🔹 Analyze one comment:</label><br>
    <textarea name="text" rows="4" placeholder="Enter a single review here..."></textarea><br><br>
    <label>🔸 Or upload file (.csv/.xlsx) with 'review' column:</label><br>
    <input type="file" name="file"><br><br>
    <input type="submit" value="Analyze">
  </form>
  {% if result %}
    <div class="result"><b>Result:</b><br>{{ result }}</div>
  {% endif %}
</body>
</html>
"""

def query_deepseek(review, refs):
    prompt = f"""You're an AI analyst for customer feedback.

Review:
"{review}"

Based on similar reviews:
- "{refs[0]}"
- "{refs[1]}"
- "{refs[2]}"

Please answer:
1. Sentiment (Positive / Negative)?
2. User feeling or attitude?
3. Pain point?
4. Suggested improvement?
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

def analyze_batch(reviews):
    sample = " ".join(reviews[:10])[:3000]  # 截取前10条合并摘要
    prompt = f"""You are an AI sentiment analyst.

Here are multiple customer reviews:
{sample}

Please summarize:
1. Overall sentiment (positive/negative/mixed)?
2. Main pain points?
3. Common expressions or emotions?
4. Suggestions to improve the product.
Answer concisely."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        file = request.files.get("file")

        if text:
            q_vec = embedder.encode([text])
            D, I = index.search(q_vec, k=3)
            references = [texts[i] for i in I[0]]
            result = query_deepseek(text, references)

        elif file:
            ext = file.filename.split(".")[-1].lower()
            try:
                if ext == "csv":
                    df = pd.read_csv(file)
                elif ext in ["xlsx", "xls"]:
                    df = pd.read_excel(file)
                else:
                    result = "❌ Unsupported file format"
                    return render_template_string(HTML, result=result)

                if "review" not in df.columns:
                    result = "❌ Please ensure your file has a 'review' column."
                else:
                    reviews = df["review"].dropna().astype(str).tolist()
                    result = analyze_batch(reviews)
            except Exception as e:
                result = f"❌ Error reading file: {str(e)}"

    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(debug=True)
