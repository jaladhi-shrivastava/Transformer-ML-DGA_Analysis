# api/app.py
import os

import joblib
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, validator
import asyncio
from functools import lru_cache

from src.config import MODELS_DIR
from src.predict import predict_fault

app = FastAPI()

@lru_cache()  # load model once, not per request
def get_model():
    return joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))

class GasInput(BaseModel):
    H2: float; CH4: float; C2H6: float
    C2H4: float; C2H2: float; CO: float; CO2: float
    model: str = "random_forest"  # allow model selection

    @validator('*', pre=True)
    def must_be_positive(cls, v):
        if isinstance(v, float) and v < 0:
            raise ValueError("Gas concentrations must be non-negative")
        return v

@app.post("/predict")
def predict(data: GasInput):
    model = get_model()
    result = predict_fault(data.dict())
    return {"fault": result, "model_used": data.model}