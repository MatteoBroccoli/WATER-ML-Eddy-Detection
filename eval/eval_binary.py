import sys
import os
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import datetime as dt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import data_utils as du
from src import eval_utils as eu

# --- CONFIGURATION ---
exp_name = str(sys.argv[1])
logit = float(sys.argv[2])
data_path = '/work/cmcc/mb31322/water/DET/AUGUST_2025/'
training_path = f'/work/cmcc/mb31322/water/unet/swot/{exp_name}/'
saving_path = training_path + f'plots/logit{logit}/'
os.makedirs(saving_path, exist_ok=True)

regions = {
    "Agulhas": [20, 30, -38, -30],
    "Lofoten": [-8, 20, 64, 74],
    "Archer": [15, 17, -35, -32.5],
    "NatalPulse": [25, 33, -35, -29],
    "SO": [-20, 50, -45, -25],
    "GS": [-80, -40, 30, 45],
}

# --- 1. DATA LOADING & PREPROCESSING ---
print('Loading SWOT validation data...')
start_val = dt.datetime(year=2023, month=7, day=27)
end_val = dt.datetime(year=2024, month=5, day=1)

ssh, _, w_exc, w_tot, _, _, _, _ = du.load_data(start_val, end_val, data_path, 'SWOT')
mask_val = np.where(ssh > 8e+36, 0, 1)
_, lat, lon = du.load_mesh()
eddy_mask_val = (w_tot + w_exc)

predictions = np.load(training_path + 'validation_predictions.npy')[..., 0]
pred_segmented = (predictions * mask_val > logit).astype(int)

print('Removing small eddies (<300km2)...')
for day in range(pred_segmented.shape[0]):
    p_struct, _ = label(pred_segmented[day], np.ones((3,3)))
    t_struct, _ = label(eddy_mask_val[day], np.ones((3,3)))
    pred_segmented[day] = np.where(eu.eddy_total_area_map(p_struct, lat, lon) > 300, pred_segmented[day], 0)
    eddy_mask_val[day] = np.where(eu.eddy_total_area_map(t_struct, lat, lon) > 300, eddy_mask_val[day], 0)

# --- 2. SNAPSHOTS (GLOBAL & REGIONAL) ---
hits = (eddy_mask_val == 1) & (pred_segmented == 1)
misses = (eddy_mask_val == 1) & (pred_segmented == 0)
false_alarms = (eddy_mask_val == 0) & (pred_segmented == 1)

composites = np.zeros_like(eddy_mask_val, dtype=np.uint8)
composites[hits] = 1
composites[misses] = 2
composites[false_alarms] = 3
composites = np.where(mask_val == 1, composites, np.nan)

cmap = ListedColormap(["lightcyan", "green", "red", "blue"])
labels = ["Background", "Hit", "Miss", "False Alarm"]

for day in range(0, min(50, len(composites)), 10):
    # Global Snapshot
    fig = plt.figure(figsize=(12, 6)); ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(lon[0,:], lat[:,0], composites[day], cmap=cmap, vmin=-0.5, vmax=3.5, rasterized=True)
    ax.coastlines(); plt.title(f'Evaluation Map Day {day}')
    plt.colorbar(im, location='bottom', shrink=0.7).set_ticks([0,1,2,3], labels=labels)
    plt.savefig(saving_path+f'composite_day_{day}.png', bbox_inches='tight')
    plt.close()

    # Regional Snapshots
    for reg, extent in regions.items():
        fig = plt.figure(figsize=(12, 6)); ax = plt.axes(projection=ccrs.Mercator())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        im = ax.pcolormesh(lon[0,:], lat[:,0], composites[day], cmap=cmap, vmin=-0.5, vmax=3.5, transform=ccrs.PlateCarree())
        ax.coastlines(); plt.title(f'{reg} Evaluation Day {day}')
        plt.colorbar(im, location='bottom', shrink=0.5).set_ticks([0,1,2,3], labels=labels)
        plt.savefig(saving_path+f'composite_{reg}_day_{day}.png', bbox_inches='tight')
        plt.close()

# --- 3. STRUCTURE-WISE METRICS & PLOTTING ---
print('Running structure-wise analysis...')
res_df = eu.structure_wise_analysis(pred_segmented, eddy_mask_val, lat, lon)

# Basic Summary Bar
summary = res_df['type'].value_counts()
total_gt = (res_df['type'].isin(['hit', 'miss', 'partial_match'])).sum()
norm_summary = {'Hit': summary.get('hit', 0)/total_gt, 'Partial Match': summary.get('partial_match', 0)/total_gt,
                'Miss': summary.get('miss', 0)/total_gt, 'False Alarm': summary.get('false_alarm', 0)/total_gt}

plt.figure(figsize=(6, 4))
colors_dict = {'hit': 'green', 'partial_match': '#FFA500', 'miss': 'red', 'false_alarm': 'blue'}
plt.bar(norm_summary.keys(), norm_summary.values(), color=['green', '#FFA500', 'red', 'blue'])
plt.title('Normalized Eddy Detection Summary'); plt.savefig(saving_path+'eddy_detection_summary.png'); plt.close()

# Helper for Histograms
def plot_perf_bars(df, col, bins, labels, title, fname, x_label):
    df_copy = df.copy()
    df_copy['bin'] = pd.cut(df_copy[col], bins=bins, labels=labels, right=False)
    sum_df = df_copy.groupby(['bin', 'type']).size().unstack(fill_value=0)
    norm_df = sum_df.div(sum_df.sum(axis=1), axis=0).reindex(columns=['hit','partial_match','miss','false_alarm'])
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    norm_df.plot(kind='bar', stacked=True, color=[colors_dict[c] for c in norm_df.columns], ax=ax1)
    
    gt_counts = sum_df[['hit', 'miss', 'partial_match']].sum(axis=1)
    ax2 = ax1.twinx()
    ax2.plot(range(len(gt_counts)), gt_counts/gt_counts.sum(), color='black', marker='o', ls='--')
    ax1.set_title(title); ax1.set_xlabel(x_label); plt.xticks(rotation=0)
    plt.savefig(saving_path + fname, bbox_inches='tight'); plt.close()

# Size, Diameter, Latitude Bars
plot_perf_bars(res_df, 'size', [0, 10, 20, 50, 100, 200, 500, 1000, np.inf], 
               ['0-10','10-20','20-50','50-100','100-200','200-500','500-1000','1000+'], 
               'Performance by Size', 'perf_size.png', 'Size (pixels)')

plot_perf_bars(res_df, 'diameter_km', [0, 15, 20, 25, 35, 50, 75, 100, 150, 200, np.inf],
               ['0-15','15-20','20-25','25-35','35-50','50-75','75-100','100-150','150-200','200+'],
               'Performance by Diameter', 'perf_diameter.png', 'Diameter (km)')

res_df['lat_bin_val'] = res_df.apply(lambda r: r['target_lat'] if r['type'] != 'false_alarm' else r['pred_lat'], axis=1)
plot_perf_bars(res_df, 'lat_bin_val', [-90, -60, -15, 0, 15, 60, 90], 
               ['60S-90S','15S-60S','0-15S','0-15N','15N-60N','60N-90N'],
               'Performance by Latitude', 'perf_lat.png', 'Latitude Zone')

# Regional Histogram
def assign_region(la, lo):
    for name, ext in regions.items():
        if ext[0] <= lo <= ext[1] and ext[2] <= la <= ext[3]: return name
    return "Other"
res_df['region'] = res_df.apply(lambda r: assign_region(r['target_lat'] if r['type'] != 'false_alarm' else r['pred_lat'], 
                                                      r['target_lon'] if r['type'] != 'false_alarm' else r['pred_lon']), axis=1)
region_df = res_df[res_df['region'] != "Other"]
plot_perf_bars(region_df, 'region', None, None, 'Performance by Region', 'perf_region.png', 'Region')

# --- 4. STRICT VS SOFT METRICS ---
tp_s = (res_df['type'] == 'hit').sum()
fn_s = (res_df['type'].isin(['miss', 'partial_match'])).sum()
fp_s = (res_df['type'] == 'false_alarm').sum()
tp_soft = tp_s + (res_df['type'] == 'partial_match').sum()
fn_soft = (res_df['type'] == 'miss').sum()

p_s = tp_s/(tp_s+fp_s); r_s = tp_s/(tp_s+fn_s); f1_s = 2*p_s*r_s/(p_s+r_s)
p_soft = tp_soft/(tp_soft+fp_s); r_soft = tp_soft/(tp_soft+fn_soft); f1_soft = 2*p_soft*r_soft/(p_soft+r_soft)

plt.figure(figsize=(7,5))
x = ['Precision', 'Recall', 'F1']
plt.bar(x, [p_s, r_s, f1_s], label='Strict')
plt.bar(x, [p_soft-p_s, r_soft-r_s, f1_soft-f1_s], bottom=[p_s, r_s, f1_s], label='Soft', alpha=0.3, hatch='//')
plt.legend(); plt.title('Metrics Comparison'); plt.savefig(saving_path+'metrics_summary.png'); plt.close()

# --- 5. PIXEL-WISE CM ---
y_true = (eddy_mask_val + mask_val).flatten()
y_pred = (pred_segmented + mask_val).flatten()
for n in ['true', 'pred']:
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2], normalize=n)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Ocean', 'Eddy'])
    disp.plot(); plt.title(f'Pixel CM ({n})'); plt.savefig(saving_path+f'pixel_cm_{n}.png'); plt.close()

print(f'Binary evaluation complete for {exp_name}')