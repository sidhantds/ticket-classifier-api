from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

# 1. Load the model once at startup
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "ticket_model.joblib"
model = joblib.load(MODEL_PATH)

# 2. Simple business rule for priority
HIGH_PRIORITY = {"PAYMENT", "REFUND", "CANCEL", "ACCOUNT"}

app = FastAPI(
    title="Support Ticket Classifier API",
    description="Classifies customer support messages into a category and priority.",
    version="1.0",
)

# 3. Input and output shapes
class Ticket(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(..., min_length=3, max_length=2000, examples=["my payment failed twice"])

class BatchRequest(BaseModel):
    tickets: list[Ticket] = Field(..., min_length=1, max_length=100)

class Prediction(BaseModel):
    text: str
    category: str
    priority: str
    confidence: float

# 4. The core logic, used by both endpoints
def classify(text: str) -> Prediction:
    probs = model.predict_proba([text])[0]
    best = probs.argmax()
    category = model.classes_[best]
    return Prediction(
        text=text,
        category=category,
        priority="high" if category in HIGH_PRIORITY else "normal",
        confidence=round(float(probs[best]), 3),
    )

# 5. Endpoints
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=Prediction)
def predict(ticket: Ticket):
    return classify(ticket.text)

@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(request: BatchRequest):
    return [classify(t.text) for t in request.tickets]