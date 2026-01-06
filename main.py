from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI(title="Chat With PDF API")

# Enable CORS for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight model - Flan-T5 Small (300MB, fast, free)
print("Loading model...")
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Use CPU
device = "cpu"
model = model.to(device)
print(f"Model loaded on {device}")


class ChatRequest(BaseModel):
    context: str
    question: str


class ChatResponse(BaseModel):
    success: bool
    answer: str = None
    error: str = None

    class Config:
        # Pydantic v1 style
        schema_extra = {
            "example": {
                "success": True,
                "answer": "The answer..."
            }
        }


@app.get("/")
def root():
    return {"status": "ok", "model": model_name}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Limit context to avoid memory issues
        context = request.context[:4000] if len(request.context) > 4000 else request.context
        question = request.question
        
        # Build prompt for Flan-T5
        prompt = f"""Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""
        
        # Tokenize and generate
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
        
        return ChatResponse(success=True, answer=answer)
        
    except Exception as e:
        return ChatResponse(success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
