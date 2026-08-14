from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import os


app = FastAPI(
    title="AI Customer Support Classifier",
    description="ML-powered customer support message classification API",
    version="1.0.0"
)


# Load trained model
model = joblib.load("classifier.joblib")


# Request model
class MessageRequest(BaseModel):
    text: str


# Health check
@app.get("/")
def home():
    return FileResponse("static/index.html")


# Prediction endpoint
@app.post("/predict")
def predict(request: MessageRequest):

    text = request.text.strip()

    if not text:
        return {
            "category": "Unknown"
        }

    prediction = model.predict([text])[0]

    return {
        "text": text,
        "category": prediction
    }