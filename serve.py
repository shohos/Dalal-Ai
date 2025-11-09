# serve.py
import os, sys, joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Ensure local import works even if started from parent dir
sys.path.append(os.path.dirname(__file__))

from pricing_model import PricingBoosterModel  # shared class

pipe: PricingBoosterModel = joblib.load("xgb_pricing_pipeline_bd_best.joblib")

app = FastAPI(title="BD Pricing Model")

class Item(BaseModel):
    category: str; brand_tier: str; condition: str; season: str
    division: str; delivery_zone: str
    seller_rating: float; stock: int; shipping_days: int; demand_index: float
    competitor_price_bdt: float; cost_bdt: float; discount_pct: float
    clicks_last_7d: int; views_last_7d: int; conversions_last_7d: int
    time_on_market_days: int
    bkash_share: float; nagad_share: float; cod_share: float; card_share: float
    is_weekend: int; is_ramadan: int; is_eid: int; is_puja: int; is_boishakh: int
    vat_included: int

class Batch(BaseModel):
    items: List[Item]

@app.post("/predict")
def predict(b: Batch):
    df = pd.DataFrame([i.dict() for i in b.items])
    preds = pipe.predict(df)
    return {"predictions": [float(round(p, 0)) for p in preds]}
