from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# HuggingFace Inference API - OpenAI compatible format
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


@app.route("/")
def root():
    return jsonify({"status": "ok", "model": "meta-llama/Llama-3.2-1B-Instruct"})


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        context = data.get("context", "")[:3000]
        question = data.get("question", "")
        
        if not question:
            return jsonify({"success": False, "error": "Question required"}), 400
        
        headers = {
            "Content-Type": "application/json"
        }
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        
        # OpenAI-compatible request format
        payload = {
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided document. Be concise."
                },
                {
                    "role": "user",
                    "content": f"Document:\n{context}\n\nQuestion: {question}"
                }
            ],
            "max_tokens": 200
        }
        
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "No answer")
            return jsonify({"success": True, "answer": answer})
        else:
            return jsonify({"success": False, "error": f"API error: {response.text}"}), 500
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
