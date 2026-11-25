import joblib
import numpy as np
import pandas as pd
from ml.config import ALL_FEATURES

# Load trained model
model = joblib.load('ml/scorer.joblib')

print('Feature order:')
for i, f in enumerate(ALL_FEATURES):
    print(f'  {i}: {f}')

print()
print('Model classes:', model.classes_)
clf = model.named_steps['clf']
print('Model type:', type(clf).__name__)

# Test with a sample feature vector (using the SHAP values as rough feature values for demo)
# Note: These are SHAP attributions, not raw features. For real inference, use actual feature values.
# Feature order: 5 KPI + 7 valuation + 3 momentum + 2 volatility + 1 beta + 2 sector = 20

# Example: construct a realistic feature vector
# KPIs (5): revenue_growth, net_margin, debt_equity, roe, current_ratio
# Valuation (7): trailing_pe_log, forward_pe_log, price_to_sales_log, peg_clipped, price_to_book_log, market_cap_log, dividend_yield
# Momentum (3): mom_3m, mom_6m, mom_12m
# Volatility (2): vol_6m, vol_12m
# Beta (1): beta_12m
# Sector (2): sector_rel_6m, sector_rel_12m

sample_features = [
    -0.0243,  # revenue_growth
    -0.0457,  # net_margin
    0.0001,   # debt_equity (near zero from SHAP)
    -0.0463,  # roe
    0.0067,   # current_ratio
    0.0007,   # trailing_pe_log
    0.0033,   # forward_pe_log
    0.0065,   # price_to_sales_log
    -0.0026,  # peg_clipped
    0.0033,   # price_to_book_log
    0.0048,   # market_cap_log
    0.0060,   # dividend_yield
    -0.0009,  # mom_3m
    0.0040,   # mom_6m
    -0.0349,  # mom_12m
    0.0034,   # vol_6m
    -0.0046,  # vol_12m
    0.0180,   # beta_12m
    0.0121,   # sector_rel_6m
    -0.0070,  # sector_rel_12m
]

X = pd.DataFrame([sample_features], columns=ALL_FEATURES)

# Predict
pred = model.predict(X)[0]
proba = model.predict_proba(X)[0]
score = proba[1] * 100  # probability of class 1 (buy)

print()
print('=' * 50)
print('PREDICTION RESULT')
print('=' * 50)
print(f'Predicted class: {pred} (1=Buy, 0=Avoid)')
print(f'Buy probability: {proba[1]:.4f} ({score:.1f}%)')
print(f'Avoid probability: {proba[0]:.4f}')

# Load calibration
import json
with open('ml/calibration.json') as f:
    cal = json.load(f)
print(f'Optimal threshold: {cal["optimal_threshold"]}')
print(f'Calibrated decision: {"BUY" if proba[1] >= cal["optimal_threshold"] else "AVOID"}')