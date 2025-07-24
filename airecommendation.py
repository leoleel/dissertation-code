import os
import re
import random
import requests
import openai
from flask import Flask, request, render_template_string
from bs4 import BeautifulSoup

# ✅ DeepSeek API 配置
client = openai.OpenAI(
    api_key="sk-18aa838cc6614a26afce435ea5f45668",  # 替换为你的 key
    base_url="https://api.deepseek.com/v1"
)
MODEL = "deepseek-chat"

# ✅ 热词抓取

def fetch_trending_keywords():
    try:
        url = "https://www2.hm.com/en_gb/new-arrivals/women/view-all.html"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        items = soup.select("a.link")
        keywords = set()
        for item in items:
            text = item.get_text(strip=True).lower()
            if any(w in text for w in ["dress", "hoodie", "jacket", "skirt", "blazer", "shirt", "coat", "pants", "trouser"]):
                keywords.add(text)
        return list(keywords)[:10] if keywords else ["streetwear", "techwear", "denim"]
    except:
        return ["streetwear", "techwear", "denim"]

# ✅ Prompt: 生成文案

def generate_email_prompt(product, keywords):
    prompt = f"""
You are a creative email copywriter for a small online clothing store called "UrbanSoul Apparel".

This shop sells trendy, affordable street fashion with unique pieces from indie brands like NorBlack NorWhite, Kotn, and Everlane. The tone should feel personal, stylish, energetic, targeting Gen Z and Millennials.

You are promoting a product category: {product}.
Current fashion keywords are: {', '.join(keywords)}

Please write a compelling promotional email in HTML format with:
- Fun and trendy intro with emotional hook
- Mention 1-2 brands available
- A time-limited discount (like 20% OFF this week)
- Call to action (CTA) with a button or link
- Keep it youthful, urban, stylish, <500 words
- Output only HTML, suitable for email campaigns
"""
    return prompt

# ✅ Prompt: 生成目标受众建议

def generate_audience_prompt(product, keywords):
    prompt = f"""
You are a fashion marketing strategist. Based on the following product category: {product} and current trending keywords: {', '.join(keywords)}, please suggest the ideal target audience.

Describe:
- Age range
- Gender focus
- Income level
- Lifestyle traits or interests
- Why this audience is a good match
Respond in 5 short bullet points.
"""
    return prompt

# ✅ HTML 页面
HTML_PAGE = """
<html>
<head>
  <title>✨ AI Fashion Email & Targeting Generator</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; background: #fefaf7; padding: 30px; }
    h2 { color: #8e44ad; }
    form { background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
    input[type=text], input[type=submit] {
      width: 100%; padding: 12px; margin-top: 12px; font-size: 16px;
    }
    .email-preview, .audience-preview {
      background: #fff; border: 1px solid #ccc; padding: 20px; margin-top: 30px; border-radius: 8px;
    }
  </style>
</head>
<body>
  <h2>🎯 AI Email & Audience Generator for Fashion Ecommerce</h2>
  <form method="POST">
    <label>Enter Product Category (e.g. hoodie, linen dress):</label>
    <input type="text" name="category" required placeholder="e.g. oversized hoodie">
    <input type="submit" value="Generate Email & Audience">
  </form>

  {% if keywords %}
    <h4>🔥 Trending Keywords:</h4>
    <ul>
      {% for kw in keywords %}<li>{{ kw }}</li>{% endfor %}
    </ul>
  {% endif %}

  {% if email %}
    <div class="email-preview">
      <h3>📧 Email Campaign Preview:</h3>
      {{ email|safe }}
    </div>
  {% endif %}

  {% if audience %}
    <div class="audience-preview">
      <h3>🧑‍🤝‍🧑 Suggested Target Audience:</h3>
      <pre>{{ audience }}</pre>
    </div>
  {% endif %}
</body>
</html>
"""

# ✅ Flask 应用
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def main():
    email_html = ""
    audience_txt = ""
    keywords = []

    if request.method == "POST":
        category = request.form.get("category")
        keywords = fetch_trending_keywords()
        sampled = random.sample(keywords, min(4, len(keywords)))

        # 📧 文案生成
        email_prompt = generate_email_prompt(category, sampled)
        email_response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": email_prompt}],
            temperature=0.7
        )
        email_html = email_response.choices[0].message.content.strip()

        # 🎯 受众生成
        audience_prompt = generate_audience_prompt(category, sampled)
        audience_response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": audience_prompt}],
            temperature=0.5
        )
        audience_txt = audience_response.choices[0].message.content.strip()

    return render_template_string(HTML_PAGE, email=email_html, audience=audience_txt, keywords=keywords)

if __name__ == "__main__":
    app.run(debug=True)
