import sys
import os
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import datetime as dt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import data_utils as du

exp_name, logit = str(sys.argv[1]), float(sys.argv[2])
data_path = '/work/cmcc/mb31322/water/DET/AUGUST_2025/'
training_path = f'/work/cmcc/mb31322/water/unet/swot/{exp_name}/'
saving_path = training_path + 'plots/regression_eval/'
os.makedirs(saving_path, exist_ok=True)

regions = {
    "Agulhas": [20, 30, -38, -30], "Lofoten": [-8, 20, 64, 74], "Archer": [15, 17, -35, -32.5],
    "NatalPulse": [25, 33, -35, -29], "SO": [-20, 50, -45, -25], "GS": [-80, -40, 30, 45],
}

print('Loading SST data...')
_, _, _, _, t_anom, t_exc, _, _ = du.load_data(dt.datetime(2023,7,27), dt.datetime(2024,5,1), data_path, 'SWOT')
gt = t_anom + t_exc
preds = np.load(training_path + 'validation_t_anom_predictions.npy')[..., 0]
_, lat, lon = du.load_mesh()

error = preds - gt
bias_map = np.nanmean(error, axis=0)
rmse_map = np.sqrt(np.nanmean(error**2, axis=0))

# Snapshots and Regional Zooms
for day in range(0, min(50, len(preds)), 10):
    # Global GT/Pred/Error Snapshot
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), subplot_kw={'projection': ccrs.PlateCarree()})
    for i, (data, title, v) in enumerate(zip([gt[day], preds[day], error[day]], ['GT', 'Pred', 'Error'], [1.0, 1.0, 0.5])):
        im = axes[i].pcolormesh(lon[0,:], lat[:,0], data, cmap='RdBu_r', vmin=-v, vmax=v, transform=ccrs.PlateCarree())
        axes[i].coastlines(); axes[i].set_title(f"{title} Day {day}")
        plt.colorbar(im, ax=axes[i], shrink=0.5)
    plt.savefig(saving_path+f'snapshot_day_{day}.png'); plt.close()

    # Regional Zooms
    for reg, extent in regions.items():
        reg_dir = os.path.join(saving_path, reg)
        os.makedirs(reg_dir, exist_ok=True)
        for data, title in zip([preds[day], gt[day]], ['pred', 'gt']):
            fig = plt.figure(figsize=(10,6)); ax = plt.axes(projection=ccrs.Mercator())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            im = ax.pcolormesh(lon[0,:], lat[:,0], data, cmap='RdBu_r', vmin=-1.0, vmax=1.0, transform=ccrs.PlateCarree())
            ax.coastlines(); plt.title(f"{reg} {title} Day {day}"); plt.colorbar(im)
            plt.savefig(os.path.join(reg_dir, f"{title}_day_{day}.png")); plt.close()

# Regional Regression Stats Table
stats = []
for reg, extent in regions.items():
    l_mask = (lon >= extent[0]) & (lon <= extent[1]) & (lat >= extent[2]) & (lat <= extent[3])
    reg_bias = np.nanmean(error[:, l_mask])
    reg_rmse = np.sqrt(np.nanmean(error[:, l_mask]**2))
    stats.append({'Region': reg, 'Bias': reg_bias, 'RMSE': reg_rmse})

stats_df = pd.DataFrame(stats)
stats_df.to_csv(saving_path + 'regional_regression_stats.csv', index=False)
print(stats_df)