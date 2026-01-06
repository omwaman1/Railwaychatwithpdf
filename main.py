from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# Use HuggingFace Inference API
HF_API_URL = "https://router.huggingface.co/hf-inference/v1/models/google/flan-t5-small"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Optional: set in Railway env vars for faster responses


@app.route("/")
def root():
    return jsonify({"status": "ok", "model": "google/flan-t5-small", "type": "huggingface-inference-api"})


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        context = data.get("context", "")[:3000]  # Limit context
        question = data.get("question", "")
        
        if not question:
            return jsonify({"success": False, "error": "Question required"}), 400
        
        prompt = f"""Answer the question based on the context.

Context: {context}

Question: {question}

Answer:"""
        
        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 200}},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get("generated_text", "No answer generated")
            else:
                answer = str(result)
            return jsonify({"success": True, "answer": answer})
        else:
            return jsonify({"success": False, "error": f"HF API error: {response.text}"}), 500
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
