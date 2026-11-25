import sys
sys.path.insert(0, r'E:\Pro AI 2\BOOM')

from ml.pipeline import build_feature_vector
from backend.app.services.ml_scorer import MLScorer
import math

kpis = {
    'revenue_growth': 1.1421,
    'net_margin': 0.5585,
    'debt_equity': 0.11,
    'roe': 0.9187,
    'current_ratio': 4.44,
}

features, metadata = build_feature_vector(kpis, ticker='NVDA')

MLScorer.load()

# Get raw probability before calibration
raw_proba = float(MLScorer._pipeline.predict_proba(features)[0, 1])
print('Raw probability: ' + str(round(raw_proba, 4)))

# Apply Platt calibration manually
calib_a = MLScorer._calib_a
calib_b = MLScorer._calib_b
calib_threshold = MLScorer._calib_threshold

print('Calibration: a=' + str(calib_a) + ', b=' + str(calib_b) + ', threshold=' + str(calib_threshold))

if calib_a is not None and calib_b is not None:
    z = calib_a * raw_proba + calib_b
    cal_proba = 1.0 / (1.0 + math.exp(-z))
    cal_proba = min(max(cal_proba, 0.001), 0.999)
    print('Calibrated probability: ' + str(round(cal_proba, 4)))

# Decision logic
threshold = calib_threshold if calib_threshold is not None else 0.5
neg_threshold = 1.0 - threshold

if cal_proba >= threshold:
    label = "Strong Buy" if cal_proba >= 0.75 else "Buy"
elif cal_proba <= neg_threshold:
    label = "Strong Avoid" if cal_proba <= 0.30 else "Avoid"
else:
    label = "Hold"

print('Threshold: ' + str(threshold) + ', Neg threshold: ' + str(neg_threshold))
print('Label: ' + label)
print('Score (cal_proba * 100): ' + str(round(cal_proba * 100, 2)))
print('Confidence: ' + str(round(abs(cal_proba - 0.5) * 200, 2)))

# Also run the official score method
score, label_official, conf, ood_info = MLScorer.score(features, kpis)
print('Official: Score=' + str(score) + ', Label=' + label_official + ', Conf=' + str(conf))