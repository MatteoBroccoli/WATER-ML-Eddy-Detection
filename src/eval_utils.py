import numpy as np
import pandas as pd
from scipy.ndimage import label

def haversine(lo1, lo2, la1, la2, degrees=True, latdep_rad=True):
    lo1, lo2, la1, la2 = map(np.asanyarray, [lo1, lo2, la1, la2])
    if degrees:
        lo1, lo2, la1, la2 = map(np.radians, [lo1, lo2, la1, la2])
    a_earth, b_earth = 6378.1370, 6356.7523
    la0 = (la1 + la2) / 2
    R = np.sqrt(((a_earth**2 * np.cos(la0))**2 + (b_earth**2 * np.sin(la0))**2) / 
                ((a_earth * np.cos(la0))**2 + (b_earth * np.sin(la0))**2)) if latdep_rad else 6371.0
    sin_lat_half = np.sin((la2 - la1) / 2)
    sin_lon_half = np.sin((lo2 - lo1) / 2)
    a = sin_lat_half**2 + np.cos(la1) * np.cos(la2) * sin_lon_half**2
    return R * 2 * np.arctan2(np.sqrt(np.abs(a)), np.sqrt(1 - np.abs(a)))

def compute_pixel_areas(lat_grid, lon_grid):
    dlat = np.abs(np.diff(lat_grid[:, 0])).mean()
    dlon = np.abs(np.diff(lon_grid[0, :])).mean()
    h = haversine(lon_grid, lon_grid, lat_grid, lat_grid + dlat)
    w = haversine(lon_grid, lon_grid + dlon, lat_grid, lat_grid)
    return h * w

def eddy_total_area_map(segmentation_array, lat_grid, lon_grid):
    pixel_areas = compute_pixel_areas(lat_grid, lon_grid)
    labels_flat = segmentation_array.ravel()
    pixel_areas_flat = pixel_areas.ravel()
    total_area_per_label = np.bincount(labels_flat, weights=pixel_areas_flat, minlength=labels_flat.max() + 1)
    total_area_per_label[0] = 0.0
    return total_area_per_label[segmentation_array]

def extract_size_metrics(row):
    area = row['target_area'] if row['type'] in ['hit', 'miss', 'partial_match'] else row['prediction_area']
    area_km2 = row['target_area_km2'] if row['type'] in ['hit', 'miss', 'partial_match'] else row['prediction_area_km2']
    return area, area_km2

def structure_wise_analysis(prediction_masks, target_masks, lat_grid, lon_grid, iou_threshold=0.5):
    pixel_areas = compute_pixel_areas(lat_grid, lon_grid)
    results = []
    for day in range(prediction_masks.shape[0]):
        pred_struct, num_pred = label(prediction_masks[day], np.ones((3,3)))
        targ_struct, num_targ = label(target_masks[day], np.ones((3,3)))
        df = pd.DataFrame({'targ': targ_struct.flatten(), 'pred': pred_struct.flatten(),
                           'lat': lat_grid.flatten(), 'lon': lon_grid.flatten(), 'p_area': pixel_areas.flatten()})
        eddy_df = df[(df['targ'] > 0) | (df['pred'] > 0)]
        t_area = eddy_df[eddy_df['targ'] > 0].groupby('targ').size()
        p_area = eddy_df[eddy_df['pred'] > 0].groupby('pred').size()
        t_area_km = eddy_df[eddy_df['targ'] > 0].groupby('targ')['p_area'].sum()
        p_area_km = eddy_df[eddy_df['pred'] > 0].groupby('pred')['p_area'].sum()
        t_centroids = eddy_df[eddy_df['targ'] > 0].groupby('targ')[['lat','lon']].mean()
        p_centroids = eddy_df[eddy_df['pred'] > 0].groupby('pred')[['lat','lon']].mean()
        overlap = pd.crosstab(eddy_df['targ'], eddy_df['pred']).drop(index=0, errors='ignore').drop(columns=0, errors='ignore')
        
        matched_preds = set()
        if not overlap.empty:
            intersection = overlap.values
            union = t_area.reindex(overlap.index).values[:,None] + p_area.reindex(overlap.columns).values[None,:] - intersection
            iou = np.where(union > 0, intersection / union, 0)
            best_p_idx = iou.argmax(axis=1)
            for i, t_id in enumerate(overlap.index):
                p_id = overlap.columns[best_p_idx[i]]
                iou_val = iou[i, best_p_idx[i]]
                d_type = 'hit' if iou_val >= iou_threshold else ('partial_match' if iou_val > 0 else 'miss')
                if d_type != 'miss': matched_preds.add(p_id)
                results.append({'type': d_type, 'iou': iou_val if d_type != 'miss' else 0,
                                'target_area': t_area[t_id], 'prediction_area': p_area[p_id] if d_type != 'miss' else 0,
                                'target_area_km2': t_area_km[t_id], 'prediction_area_km2': p_area_km[p_id] if d_type != 'miss' else 0,
                                'target_lat': t_centroids.loc[t_id, 'lat'], 'target_lon': t_centroids.loc[t_id, 'lon'],
                                'pred_lat': p_centroids.loc[p_id, 'lat'] if d_type != 'miss' else None,
                                'pred_lon': p_centroids.loc[p_id, 'lon'] if d_type != 'miss' else None})
        
        for p_id in (set(p_area.index) - matched_preds):
            results.append({'type': 'false_alarm', 'iou': 0, 'target_area': 0, 'prediction_area': p_area[p_id],
                            'target_area_km2': 0, 'prediction_area_km2': p_area_km[p_id],
                            'target_lat': None, 'target_lon': None, 'pred_lat': p_centroids.loc[p_id, 'lat'], 'pred_lon': p_centroids.loc[p_id, 'lon']})
    
    res_df = pd.DataFrame(results)
    res_df['size'], res_df['size_km2'] = zip(*res_df.apply(extract_size_metrics, axis=1))
    res_df['diameter_km'] = 2 * np.sqrt(res_df['size_km2'] / np.pi)
    return res_df