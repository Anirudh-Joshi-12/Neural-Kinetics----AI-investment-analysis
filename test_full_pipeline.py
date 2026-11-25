import sys
sys.path.insert(0, '.')

from ml.pipeline import build_feature_vector
from backend.app.services.ml_scorer import MLScorer
import math

# NVIDIA FY2025 actual KPIs (from 10-K)
kpis = {
    'revenue_growth': 1.1421,   # 114.21%
    'net_margin': 0.5585,       # 55.85%
    'debt_equity': 0.11,        # 0.11
    'roe': 0.9187,              # 91.87%
    'current_ratio': 4.44,      # 4.44
}

features, metadata = build_feature_vector(kpis, ticker='NVDA')

MLScorer.load()

# Get raw probability
raw_proba = float(MLScorer._pipeline.predict_proba(features)[0, 1])
print('Raw probability: {:.4f}'.format(raw_proba))

# Apply Platt calibration manually
calib_a = MLScorer._calib_a
calib_b = MLScorer._calib_b
calib_threshold = MLScorer._calib_threshold

print('Calibration: a={}, b={}, threshold={}'.format(calib_a, calib_b, calib_threshold))

if calib_a is not None and calib_b is not None:
    z = calib_a * raw_proba + calib_b
    cal_proba = 1.0 / (1.0 + math.exp(-z))
    cal_proba = min(max(cal_proba, 0.001), 0.999)
    print('Calibrated probability: {:.4f}'.format(cal_proba))

# Decision logic
threshold = calib_threshold if calib_threshold is not None else 0.5
neg_threshold = 1.0 - threshold

if cal_proba >= threshold:
    label = "Strong Buy" if cal_proba >= 0.75 else "Buy"
elif cal_proba <= neg_threshold:
    label = "Strong Avoid" if cal_proba <= 0.30 else "Avoid"
else:
    label = "Hold"

print('Threshold: {}, Neg threshold: {}'.format(threshold, neg_threshold))
print('Label: {}'.format(label))
print('Score (cal_proba * 100): {:.2f}'.format(cal_proba * 100))
print('Confidence: {:.2f}'.format(abs(cal_proba - 0.5) * 200))

# Also run the official score method
score, label_official, conf, ood_info = MLScorer.score(features, kpis)
print('\nOfficial: Score={}, Label={}, Conf={}'.format(score, label_official, conf))
print('OOD info: {}'.format(ood_info))