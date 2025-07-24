import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# 原始数据路径
csv_path = r"D:\airesume\restaurant_qa_10k.csv"

# 读取并清洗
df = pd.read_csv(csv_path)
df = df.dropna(subset=["Question", "Answer"]).drop_duplicates()

# 生成嵌入
model = SentenceTransformer("all-MiniLM-L6-v2")
questions = df["Question"].tolist()
embeddings = model.encode(questions, convert_to_numpy=True)

# 保存向量和清洗数据
np.save(r"D:\airesume\steakhouse_q_embeddings.npy", embeddings)
df.to_csv(r"D:\airesume\steakhouse_qa_cleaned.csv", index=False)

print("✅ 嵌入文件与清洗CSV已保存至 D:\\airesume")
import os
import faiss
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, session
from sentence_transformers import SentenceTransformer
import openai

# ✅ DeepSeek 配置
client = openai.OpenAI(
    api_key="sk-18aa838cc6614a26afce435ea5f45668",  # 替换为你的真实key
    base_url="https://api.deepseek.com/v1"
)
MODEL = "deepseek-chat"

# ✅ 餐厅描述（系统知识）
restaurant_profile = """
Welcome to The Iron Flame Steakhouse — a family-friendly restaurant located at 123 Oak Street, Dublin.

🕒 Opening Hours: Monday to Saturday, 12 PM - 11 PM (Closed on Sundays)
📞 Phone: +353 123 4567

🍽️ Services:
- Dine-in, takeaway, and online ordering
- Signature dishes: Dry-aged Ribeye, Garlic Butter Filet Mignon, Wagyu Sliders
- Homemade desserts, house red wine
- Limited vegetarian options
- Accepts card payments and Apple Pay
- Wheelchair accessible and reservation supported
"""

# ✅ 数据路径（确保与预处理脚本一致）
embedding_path = r"D:\airesume\steakhouse_q_embeddings.npy"
data_path = r"D:\airesume\steakhouse_qa_cleaned.csv"
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ 加载数据
qa_df = pd.read_csv(data_path)
questions = qa_df["Question"].tolist()
answers = qa_df["Answer"].tolist()
embeddings = np.load(embedding_path)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

app = Flask(__name__)
app.secret_key = "ironflame-secret"

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>The Iron Flame Chatbot</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; margin: 40px; background: #fff9f3; color: #333; }
    h2 { color: #800000; }
    textarea, button { font-size: 16px; padding: 10px; width: 100%; margin-top: 10px; }
    .result, .history { background: #f2f2f2; padding: 15px; border-radius: 8px; margin-top: 20px; white-space: pre-wrap; }
    footer { margin-top: 60px; font-size: 14px; color: #888; }
  </style>
</head>
<body>
  <h2>🥩 Welcome to The Iron Flame Steakhouse Assistant</h2>
  <p>Ask anything about our restaurant. Chat history will be remembered.</p>

  <form method="POST">
    <textarea name="question" rows="4" placeholder="e.g. What steak do you recommend?" required></textarea>
    <button type="submit">Ask</button>
  </form>

  {% if history %}
    <div class="history">
      <strong>🗨️ Conversation History:</strong><br>
      {{ history }}
    </div>
  {% endif %}

  {% if answer %}
    <div class="result">
      <strong>🤖 Bot:</strong><br>
      {{ answer }}
    </div>
  {% endif %}

  <footer>
    The Iron Flame Steakhouse · 123 Oak Street, Dublin · Open Mon-Sat 12PM-11PM · Closed Sunday
  </footer>
</body>
</html>
"""

# ✅ fallback：结合上下文使用 DeepSeek 回答
def fallback_deepseek_conversation(chat_history):
    messages = [
        {"role": "system", "content": f"You are an assistant for The Iron Flame Steakhouse. Use only this info to answer:\n\n{restaurant_profile}"}
    ] + chat_history

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

@app.route("/", methods=["GET", "POST"])
def chatbot():
    if "chat_history" not in session:
        session["chat_history"] = []

    answer = ""
    display_history = ""

    if request.method == "POST":
        question = request.form.get("question")
        q_vec = model.encode([question])
        D, I = index.search(q_vec, k=1)
        matched_answer = answers[I[0][0]]
        similarity_score = D[0][0]

        if similarity_score > 1.0:
            session["chat_history"].append({"role": "user", "content": question})
            reply = fallback_deepseek_conversation(session["chat_history"])
            session["chat_history"].append({"role": "assistant", "content": reply})
            answer = reply
        else:
            answer = matched_answer
            session["chat_history"].append({"role": "user", "content": question})
            session["chat_history"].append({"role": "assistant", "content": answer})

        display_history = "\n".join(
            ["👤 " + msg["content"] if msg["role"] == "user" else "🤖 " + msg["content"] for msg in session["chat_history"][-6:]]
        )

    return render_template_string(HTML_PAGE, answer=answer, history=display_history)

if __name__ == "__main__":
    app.run(debug=True)