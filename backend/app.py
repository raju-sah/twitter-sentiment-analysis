import pickle
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nltk_setup import ensure_nltk_data
from utils import predict_tweet, process_tweet

MODEL_PATH = Path(__file__).parent / "model.pkl"

ensure_nltk_data()

with open(MODEL_PATH, "rb") as f:
    _model = pickle.load(f)
FREQS = _model["freqs"]
THETA = _model["theta"]


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    probability: float
    processed_tokens: List[str]


app = FastAPI(title="Twitter Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    prob = predict_tweet(req.text, FREQS, THETA)
    label = "positive" if prob > 0.5 else "negative"
    return PredictResponse(
        label=label,
        probability=prob,
        processed_tokens=process_tweet(req.text),
    )
