from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)
CORS(app)

# Load lightweight model - Flan-T5 Small (300MB, fast, free)
print("Loading model...")
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cpu"
model = model.to(device)
print(f"Model loaded on {device}")


@app.route("/")
def root():
    return jsonify({"status": "ok", "model": model_name})


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        context = data.get("context", "")[:4000]
        question = data.get("question", "")
        
        if not question:
            return jsonify({"success": False, "error": "Question required"}), 400
        
        prompt = f"""Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({"success": True, "answer": answer})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
