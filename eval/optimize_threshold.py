import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from scipy.ndimage import label

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import data_utils as du
from src import eval_utils as eu

exp_name = str(sys.argv[1])
data_path = '/work/cmcc/mb31322/water/DET/AUGUST_2025/'
training_path = f'/work/cmcc/mb31322/water/unet/swot/{exp_name}/'
saving_path = training_path + f'plots/'
os.makedirs(saving_path, exist_ok=True)

# Validation Dates
end_date = dt.datetime(year=2024, month=5, day=1)
start_val = dt.datetime(year=2023, month=7, day=27)

print('Loading data for optimization...')
preds = np.load(training_path + 'validation_predictions.npy')[..., 0]
ssh, _, w_exc, w_tot, _, _, _, _ = du.load_data(start_val, end_date, data_path, 'SWOT')
mask = np.where(ssh > 8e+36, 0, 1)
_, lat, lon = du.load_mesh()

target = w_tot + w_exc
tmp_days = 30 # Following original script logic
tmp_logit = preds[:tmp_days] * mask[:tmp_days]
tmp_targ = target[:tmp_days]

logit_thresholds = np.linspace(-2.5, 2.5, 25)
precisions, recalls, f1_scores = [], [], []

print(f'Running optimization over {len(logit_thresholds)} thresholds...')
for logit_thresh in logit_thresholds:
    p_bin = (tmp_logit > logit_thresh).astype(int)
    t_bin = tmp_targ.copy()
    
    # Remove small eddies < 300km2
    for day in range(p_bin.shape[0]):
        p_struct, _ = label(p_bin[day], np.ones((3,3)))
        t_struct, _ = label(t_bin[day], np.ones((3,3)))
        p_bin[day] = np.where(eu.eddy_total_area_map(p_struct, lat, lon) > 300, p_bin[day], 0)
        t_bin[day] = np.where(eu.eddy_total_area_map(t_struct, lat, lon) > 300, t_bin[day], 0)

    res_df = eu.structure_wise_analysis(p_bin, t_bin, lat, lon)
    
    tp = (res_df['type'] == 'hit').sum()
    fn = (res_df['type'].isin(['miss', 'partial_match'])).sum()
    fp = (res_df['type'] == 'false_alarm').sum()

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    precisions.append(p); recalls.append(r); f1_scores.append(f1)

best_idx = np.argmax(f1_scores)
best_thresh = logit_thresholds[best_idx]
print(f"Optimal threshold: {best_thresh:.3f} (F1: {f1_scores[best_idx]:.3f})")

plt.figure(figsize=(10, 6))
plt.plot(logit_thresholds, f1_scores, label='F1', linewidth=2)
plt.plot(logit_thresholds, precisions, '--', label='Precision')
plt.plot(logit_thresholds, recalls, ':', label='Recall')
plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best Thresh: {best_thresh:.2f}')
plt.xlabel('Logit Threshold'); plt.ylabel('Score'); plt.grid(True); plt.legend()
plt.savefig(saving_path+'scores_vs_logit_threshold.png')

with open(saving_path + "best_threshold_f1.txt", "w") as f:
    f.write(f"{best_thresh:.3f}")