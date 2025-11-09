# pricing_model.py
from typing import List
import pandas as pd
from scipy import sparse
import xgboost as xgb

class PricingBoosterModel:
    """
    Lightweight pipeline wrapper:
      - fe: feature engineer (Transformer)
      - pre: one-hot encoder (ColumnTransformer)
      - booster: xgboost.Booster
    """
    def __init__(self, fe, pre, booster, num_feature_names: List[str], cat_cols: List[str]):
        self.fe = fe
        self.pre = pre
        self.booster = booster
        self.num_feature_names = num_feature_names
        self.cat_cols = cat_cols

    def _transform(self, Xdf: pd.DataFrame) -> xgb.DMatrix:
        Xf = self.fe.transform(Xdf)
        Xenc = self.pre.transform(Xf)
        if not sparse.isspmatrix_csr(Xenc):
            from scipy import sparse as sp
            Xenc = sp.csr_matrix(Xenc)
        return xgb.DMatrix(Xenc)

    def predict(self, Xdf: pd.DataFrame):
        dm = self._transform(Xdf)
        return self.booster.predict(dm)
