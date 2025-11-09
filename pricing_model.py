# pricing_model.py
from typing import List
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb

class FeatureEngineer:
    """Stateless FE used in both training & serving."""
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "competitor_price_bdt" in X and "cost_bdt" in X:
            X["gap_comp_cost"] = X["competitor_price_bdt"] - X["cost_bdt"]
            X["ratio_comp_cost"] = (X["competitor_price_bdt"] + 1e-9) / (X["cost_bdt"] + 1e-9)
        if "clicks_last_7d" in X and "views_last_7d" in X:
            X["ctr_7d"] = np.where(X["views_last_7d"] > 0, X["clicks_last_7d"] / X["views_last_7d"], 0.0)
        if "conversions_last_7d" in X and "clicks_last_7d" in X:
            X["cr_7d"] = np.where(X["clicks_last_7d"] > 0, X["conversions_last_7d"] / X["clicks_last_7d"], 0.0)
        need = ["bkash_share","nagad_share","cod_share"]
        if all(c in X for c in need):
            X["prepaid_share"] = X["bkash_share"] + X["nagad_share"]
            X["cash_heavy"] = (X["cod_share"] > 0.5).astype(int)
        # extras that helped tuning
        if "competitor_price_bdt" in X and "cost_bdt" in X:
            X["margin_room"] = X["competitor_price_bdt"] - X["cost_bdt"]
            X["margin_ratio"] = (X["competitor_price_bdt"] + 1e-9) / (X["cost_bdt"] + 1e-9)
        return X

class PricingBoosterModel:
    """Lightweight pipeline wrapper for FE + OneHot + Booster."""
    def __init__(self, fe, pre, booster, num_feature_names: List[str], cat_cols: List[str]):
        self.fe = fe
        self.pre = pre
        self.booster = booster
        self.num_feature_names = num_feature_names
        self.cat_cols = cat_cols

    def _to_dmatrix(self, Xdf: pd.DataFrame) -> xgb.DMatrix:
        Xf = self.fe.transform(Xdf)
        Xenc = self.pre.transform(Xf)
        if not sparse.isspmatrix_csr(Xenc):
            from scipy import sparse as sp
            Xenc = sp.csr_matrix(Xenc)
        return xgb.DMatrix(Xenc)

    def predict(self, Xdf: pd.DataFrame):
        dm = self._to_dmatrix(Xdf)
        return self.booster.predict(dm)
