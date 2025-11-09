# migrate_model.py
import sys, types, joblib
from pricing_model import PricingBoosterModel as NewPBM, FeatureEngineer as NewFE

# ---- Stubs so pickle can load the OLD file ----
class PricingBoosterModel:  # old path: __main__.PricingBoosterModel
    def __init__(self, fe, pre, booster, num_feature_names, cat_cols):
        self.fe = fe
        self.pre = pre
        self.booster = booster
        self.num_feature_names = num_feature_names
        self.cat_cols = cat_cols

class FeatureEngineer:      # old path: __main__.FeatureEngineer
    def __init__(self): pass

# Ensure stubs are visible as __main__.* for unpickling
if '__main__' not in sys.modules:
    sys.modules['__main__'] = types.ModuleType('__main__')
setattr(sys.modules['__main__'], 'PricingBoosterModel', PricingBoosterModel)
setattr(sys.modules['__main__'], 'FeatureEngineer', FeatureEngineer)

# ---- Load old artifact, rebuild with real classes, re-save ----
old = joblib.load("xgb_pricing_pipeline_bd_best.joblib")

# Replace FE with the real implementation (training FE was stateless)
fe = NewFE()
pre = old.pre
booster = old.booster
num_feature_names = getattr(old, 'num_feature_names', [])
cat_cols = getattr(old, 'cat_cols', [])

new = NewPBM(fe=fe, pre=pre, booster=booster,
             num_feature_names=num_feature_names, cat_cols=cat_cols)

joblib.dump(new, "xgb_pricing_pipeline_bd_best.joblib")
print("✅ Re-saved with module path: pricing_model.PricingBoosterModel (and real FeatureEngineer).")
