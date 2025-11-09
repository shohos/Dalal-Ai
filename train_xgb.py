# train_xgb_bd_tuned.py  (Booster-based, no sklearn .fit kwargs)
from pricing_model import PricingBoosterModel
import sys, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
import joblib
from pprint import pprint

def booster_predict_best(booster, dmat):
    """Predict using early-stopped best iteration if available; robust across xgboost variants."""
    # Preferred: use best_iteration with iteration_range
    bi = getattr(booster, "best_iteration", None)
    if bi is not None:
        try:
            return booster.predict(dmat, iteration_range=(0, bi + 1))
        except TypeError:
            pass  # older builds may not support iteration_range

    # Fallback: try best_ntree_limit (some older builds expose this)
    ntl = getattr(booster, "best_ntree_limit", None)
    if ntl:
        try:
            return booster.predict(dmat, ntree_limit=ntl)
        except TypeError:
            pass

    # Last resort: use all boosted rounds
    try:
        rounds = booster.num_boosted_rounds()
        return booster.predict(dmat, iteration_range=(0, rounds))
    except Exception:
        # Plain predict
        return booster.predict(dmat)


RANDOM_STATE = 42

# ---------------- Feature engineering ----------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if "competitor_price_bdt" in X and "cost_bdt" in X:
            X["gap_comp_cost"] = X["competitor_price_bdt"] - X["cost_bdt"]
            X["ratio_comp_cost"] = (X["competitor_price_bdt"] + 1e-9) / (X["cost_bdt"] + 1e-9)
        if "clicks_last_7d" in X and "views_last_7d" in X:
            X["ctr_7d"] = np.where(X["views_last_7d"]>0, X["clicks_last_7d"]/X["views_last_7d"], 0.0)
        if "conversions_last_7d" in X and "clicks_last_7d" in X:
            X["cr_7d"] = np.where(X["clicks_last_7d"]>0, X["conversions_last_7d"]/X["clicks_last_7d"], 0.0)
        need = ["bkash_share","nagad_share","cod_share"]
        if all(c in X for c in need):
            X["prepaid_share"] = X["bkash_share"] + X["nagad_share"]
            X["cash_heavy"] = (X["cod_share"] > 0.5).astype(int)
        return X

# ---------------- Lightweight wrapper we can persist ----------------
class PricingBoosterModel:
    """
    A tiny 'pipeline' that stores:
      - feature engineer (FE)
      - one-hot encoder (pre)
      - xgboost Booster (core model)
      - list of numeric feature names after FE
    Provides .predict(df) -> np.array
    """
    def __init__(self, fe, pre, booster, num_feature_names, cat_cols):
        self.fe = fe
        self.pre = pre
        self.booster = booster
        self.num_feature_names = num_feature_names
        self.cat_cols = cat_cols

    def _transform(self, Xdf):
        Xf = self.fe.transform(Xdf)
        Xenc = self.pre.transform(Xf)
        # ensure CSR for xgboost
        if not sparse.isspmatrix_csr(Xenc):
            Xenc = sparse.csr_matrix(Xenc)
        return xgb.DMatrix(Xenc)

    def predict(self, Xdf):
        dm = self._transform(Xdf)
        return self.booster.predict(dm)

# ---------------- Data loading ----------------
def load_data(path="price_training_dataset_bd.csv"):
    df = pd.read_csv(path)
    num_cols = [c for c in df.columns if c.endswith("_bdt") or c.endswith("_share") or c in [
        "seller_rating","stock","shipping_days","demand_index","discount_pct",
        "clicks_last_7d","views_last_7d","conversions_last_7d","time_on_market_days",
        "is_weekend","is_ramadan","is_eid","is_puja","is_boishakh","vat_included",
        "target_price_bdt"
    ]]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df

def sample_params(rng):
    return {
        "max_depth": int(rng.integers(6, 14)),          # was 6..11
        "eta": float(rng.choice([0.02, 0.025, 0.03, 0.035, 0.04])),
        "subsample": float(rng.uniform(0.6, 0.95)),
        "colsample_bytree": float(rng.uniform(0.6, 0.95)),
        "min_child_weight": float(rng.choice([1, 2, 3, 4, 5, 6])),
        "alpha": float(rng.choice([0.0, 0.5, 1.0, 2.0, 4.0])),
        "lambda": float(rng.choice([1.0, 2.0, 3.0, 5.0, 8.0])),
        "gamma": float(rng.choice([0, 0.5, 1.0])),
        "objective": str(np.random.choice(["reg:absoluteerror","reg:squarederror"])),
        # Optional extra knobs:
        "max_bin": int(rng.integers(128, 512)),         # helps on sparse one-hot
    }

def main():
    df = load_data("price_training_dataset_bd.csv")
    y = df["target_price_bdt"]
    X = df.drop(columns=["target_price_bdt","product_id"])

    # 70/15/15
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1765, random_state=RANDOM_STATE)

    # FE + OneHot
    cat_cols = ["category","brand_tier","condition","season","division","delivery_zone"]
    fe = FeatureEngineer()
    Xt = fe.fit_transform(X_train)
    Xv = fe.transform(X_val)
    Xte= fe.transform(X_test)

    num_cols = [c for c in Xt.columns if c not in cat_cols]
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])
    pre.fit(Xt)

    Xtr_enc = pre.transform(Xt)
    Xval_enc= pre.transform(Xv)

    # to DMatrix
    if not sparse.isspmatrix_csr(Xtr_enc): Xtr_enc = sparse.csr_matrix(Xtr_enc)
    if not sparse.isspmatrix_csr(Xval_enc): Xval_enc = sparse.csr_matrix(Xval_enc)
    dtrain = xgb.DMatrix(Xtr_enc, label=y_train.values)
    dval   = xgb.DMatrix(Xval_enc, label=y_val.values)

    # Random search with early stopping via xgb.train
    rng = np.random.default_rng(RANDOM_STATE)
    trials = 120
    best = None

    for i in range(1, trials+1):
        p = sample_params(rng)
        params = {
            "max_depth": p["max_depth"],
            "eta": p["eta"],
            "subsample": p["subsample"],
            "colsample_bytree": p["colsample_bytree"],
            "min_child_weight": p["min_child_weight"],
            "alpha": p["alpha"],
            "lambda": p["lambda"],
            "gamma": p["gamma"],
            "objective": p["objective"],
            "tree_method": "hist",
            "eval_metric": "mae",
            "seed": RANDOM_STATE
        }
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=8000,        # was 4000
            evals=[(dval, "val")],
            early_stopping_rounds=300,   # was 150
            verbose_eval=False
        )
        # Evaluate on val using best ntree limit
        pred_val = booster_predict_best(booster, dval)
        mae = mean_absolute_error(y_val, pred_val)
        entry = {"mae": mae, "best_iteration": booster.best_iteration, "params": params}
        if (best is None) or (mae < best["mae"]): best = entry
        if i % 5 == 0:
            print(f"[{i}/{trials}] best MAE: {best['mae']:.0f} @ iter {best['best_iteration']}")

    print("\n=== BEST CONFIG ===")
    pprint({k:(v if k!='params' else {kk:vv for kk,vv in v.items() if kk!='seed'}) for k,v in best.items()})

    # Refit on Train+Val with best_iteration
    Xtrv = pd.concat([Xt, Xv], axis=0, ignore_index=True)
    ytrv = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    for _df in (Xt, Xv, Xte):
        if {"competitor_price_bdt","cost_bdt"}.issubset(_df.columns):
            _df["margin_room"] = _df["competitor_price_bdt"] - _df["cost_bdt"]
            _df["margin_ratio"] = (_df["competitor_price_bdt"] + 1e-9) / (_df["cost_bdt"] + 1e-9)
    pre2 = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ]).fit(Xtrv)

    Xtrv_enc = pre2.transform(Xtrv)
    if not sparse.isspmatrix_csr(Xtrv_enc): Xtrv_enc = sparse.csr_matrix(Xtrv_enc)
    dtrv = xgb.DMatrix(Xtrv_enc, label=ytrv.values)

    booster_final = xgb.train(
        params=best["params"],
        dtrain=dtrv,
        num_boost_round=int(best["best_iteration"] or 2500),
        evals=[],
        verbose_eval=False
    )

    # Test evaluation
    Xte_enc = pre2.transform(Xte)
    if not sparse.isspmatrix_csr(Xte_enc): Xte_enc = sparse.csr_matrix(Xte_enc)
    dtest = xgb.DMatrix(Xte_enc, label=y_test.values)
    pred_test = booster_final.predict(dtest)
    mae_test = mean_absolute_error(y_test, pred_test)
    within_500 = (np.abs(pred_test - y_test) <= 500).mean()*100
    within_1000 = (np.abs(pred_test - y_test) <= 1000).mean()*100

    print("\n=== TEST METRICS ===")
    print(f"Test MAE (BDT): {mae_test:.0f}")
    print(f"Within ±৳500:  {within_500:.1f}%")
    print(f"Within ±৳1,000:{within_1000:.1f}%")

    # Save lightweight pipeline (FE + OneHot + Booster)
    pipe = PricingBoosterModel(
    fe=fe, pre=pre2, booster=booster_final,
    num_feature_names=[c for c in Xtrv.columns if c not in cat_cols],
    cat_cols=cat_cols
    )
    joblib.dump(pipe, "xgb_pricing_pipeline_bd_best.joblib")
    print("Saved to xgb_pricing_pipeline_bd_best.joblib")

if __name__ == "__main__":
    main()
