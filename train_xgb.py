# train_xgb.py
# Minimal XGBoost regression example for the synthetic pricing dataset
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

df = pd.read_csv('price_training_dataset.csv')

y = df['target_price']
X = df.drop(columns=['target_price','product_id'])

cat_cols = ['category','brand_tier','condition','season','region']
num_cols = [c for c in X.columns if c not in cat_cols]

pre = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])


model = xgb.XGBRegressor(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=3.0,
    tree_method='hist',
    random_state=42
)

pipe = Pipeline([('pre', pre), ('xgb', model)])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe.fit(
    X_train, y_train,
    xgb__eval_set=[(pre.fit_transform(X_val), y_val)],
    xgb__verbose=False,
    xgb__early_stopping_rounds=100
)
pred = pipe.predict(X_val)
mae = mean_absolute_error(y_val, pred)

print(f"Validation MAE: {mean_absolute_error(y_val, pred):.2f}")
# Save the trained pipeline (preprocessing + model)
import joblib
joblib.dump(pipe, 'xgb_pricing_pipeline.joblib')
print('Saved to xgb_pricing_pipeline.joblib')
