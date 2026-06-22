"""
detection_utils.py
==================
Self-contained helper module for the VarDyn eddy detection pipeline.

Contains:
  - U-Net model definition  (frozen copy from src/model.py)
  - Loss / metric functions (frozen copy from src/losses.py)
  - Data I/O  (mesh, norm params, VarDyn)
  - Pre-processing / embedding
  - Geometry  (haversine, pixel areas)
  - Post-processing  (segmentation, active/inactive classification)
  - Output writing

NOTE: The model architecture and loss functions are intentionally frozen here
and must NOT be modified. Any change would break compatibility with the
pre-trained checkpoint weights.
"""

import os
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import label
from scipy.ndimage import maximum as nd_maximum

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# Model architecture  (frozen from src/model.py)
# =============================================================================

def get_unet(input_shape, output_channels=1, embed_dim=64, num_stages=3,
             kernel_size=4, strides=2):
    """Standard U-Net architecture used for WATER eddy detection and SST reconstruction."""
    inputs = keras.Input(shape=input_shape)
    x_downsample = []
    x = inputs
    x_downsample.append(x)

    # Encoder
    for stage in range(num_stages):
        initializer = tf.random_normal_initializer(0., 0.02)
        x = layers.Conv2D(embed_dim * 2**stage, kernel_size, strides,
                          activation='relu', padding='same',
                          kernel_initializer=initializer, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x_downsample.append(x)

    # Decoder
    x_downsample = list(reversed(x_downsample[:-1]))
    for stage in range(num_stages - 1):
        initializer = tf.random_normal_initializer(0., 0.02)
        x = layers.Conv2DTranspose(embed_dim * 2**(num_stages - stage - 2),
                                   kernel_size, strides, activation='relu',
                                   padding='same',
                                   kernel_initializer=initializer,
                                   use_bias=False)(x)
        x = layers.Resizing(height=x_downsample[stage].shape[1],
                            width=x_downsample[stage].shape[2])(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Concatenate()([x, x_downsample[stage]])

    # Output
    initializer = tf.random_normal_initializer(0., 0.02)
    outputs = layers.Conv2DTranspose(output_channels, kernel_size, strides,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     activation=None)(x)
    outputs = layers.Resizing(height=x_downsample[-1].shape[1],
                              width=x_downsample[-1].shape[2])(outputs)

    return keras.Model(inputs=inputs, outputs=outputs)


# =============================================================================
# Loss and metric functions  (frozen from src/losses.py)
# =============================================================================

def detection_metric(y_true, y_pred):
    """BCE for Eddy Mask (Channel 0)."""
    det_true = y_true[..., 0:1]
    det_pred = y_pred[..., 0:1]
    return tf.keras.losses.binary_crossentropy(det_true, det_pred, from_logits=True)


def temp_metric(y_true, y_pred):
    """MSE for Temperature Anomaly (Channel 1)."""
    temp_true = y_true[..., 1:2]
    temp_pred = y_pred[..., 1:2]
    return tf.keras.losses.mean_squared_error(temp_true, temp_pred)


def boundary_sst_coupling_loss(y_true, y_pred, branch_align=False):
    """Custom loss to force spatial agreement between mask and SST borders."""
    mask_true        = y_true[..., 0:1]
    sst_true         = y_true[..., 1:2]
    mask_pred_prob   = tf.math.sigmoid(y_pred[..., 0:1])
    sst_pred         = y_pred[..., 1:2]

    ksize = 5
    inv_mask_true  = 1.0 - mask_true
    eroded_true    = 1.0 - tf.nn.max_pool2d(inv_mask_true, ksize=ksize, strides=1, padding='SAME')
    boundary_zone_true = mask_true - eroded_true

    inv_mask_pred  = 1.0 - mask_pred_prob
    eroded_pred    = 1.0 - tf.nn.max_pool2d(inv_mask_pred, ksize=ksize, strides=1, padding='SAME')
    boundary_zone_pred = mask_pred_prob - eroded_pred

    target_product    = boundary_zone_true * sst_true
    predicted_product = boundary_zone_pred * sst_pred

    product_mse = tf.reduce_mean(tf.square(target_product - predicted_product))
    if branch_align:
        border_alignment = tf.reduce_mean(tf.square(boundary_zone_true - boundary_zone_pred))
        product_mse += 0.1 * border_alignment
    return product_mse


def gradient_loss(y_true, y_pred):
    """Ensures sharp temperature boundaries."""
    dy_true, dx_true = tf.image.image_gradients(y_true[..., 1:2])
    dy_pred, dx_pred = tf.image.image_gradients(y_pred[..., 1:2])
    return tf.reduce_mean(tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true))


def combined_eddy_loss(y_true, y_pred):
    bce      = detection_metric(y_true, y_pred)
    mse      = temp_metric(y_true, y_pred)
    coupling = boundary_sst_coupling_loss(y_true, y_pred, branch_align=True)
    grad     = gradient_loss(y_true, y_pred)
    return bce + (1000 * mse) + (1000 * coupling) + grad


# =============================================================================
# I/O helpers
# =============================================================================

def load_mesh(mesh_dir):
    """Load land-sea mask and coordinate arrays from meshmask.nc."""
    ds  = Dataset(os.path.join(mesh_dir, "meshmask.nc"))
    mask = np.array(ds.variables["lsm"])
    lat  = np.array(ds.variables["lat"])
    lon  = np.array(ds.variables["lon"])
    ds.close()
    return mask, lat, lon


def load_norm_params(norm_params_path):
    """Load normalisation parameters saved during training."""
    with np.load(norm_params_path) as data:
        params = {
            "ssh_maxabs":    data["ssh_maxabs"],
            "sst_maxabs":    data["sst_maxabs"],
            "ugosa_maxabs":  data["ugosa_maxabs"],
            "vgosa_maxabs":  data["vgosa_maxabs"],
            "sst_mean":      data["sst_reference_mean"],
            "phase1_maxabs": data["phase1_maxabs"],
            "phase2_maxabs": data["phase2_maxabs"],
            "t_anom_maxabs": data["t_anom_maxabs"],
        }
    return params


def load_vardyn(vardyn_file):
    """Load all variables from a VarDyn NetCDF file."""
    ds   = Dataset(vardyn_file)
    sla  = np.array(ds.variables["sla"])
    sst  = np.array(ds.variables["sst"])
    ugos = np.array(ds.variables["ugos"])
    vgos = np.array(ds.variables["vgos"])
    lat  = np.array(ds.variables["latitude"])
    lon  = np.array(ds.variables["longitude"])
    time = np.array(ds.variables["time"])
    ds.close()
    return sla, sst, ugos, vgos, lat, lon, time


# =============================================================================
# Pre-processing
# =============================================================================

FILL_VALUE_THRESHOLD = -1e8


def mask_fill_values(*arrays):
    """Replace fill values with NaN in all input arrays; return a tuple."""
    return tuple(
        np.where(arr < FILL_VALUE_THRESHOLD, np.nan, arr)
        for arr in arrays
    )


def kelvin_to_celsius(sst):
    """Convert SST from Kelvin to Celsius and mask physically impossible values."""
    sst = sst - 273.15
    return np.where(sst < -2, np.nan, sst)


def embed_regional_data(regional_arrays, lat_global, lon_global,
                        lat_regional, lon_regional, n_rows, n_cols):
    """
    Embed regional (subdomain) arrays into a global-sized (n_rows x n_cols)
    array filled with NaN.

    Parameters
    ----------
    regional_arrays : list of ndarray  (time, lat_r, lon_r)
    lat_global, lon_global : 2-D global coordinate grids
    lat_regional, lon_regional : 1-D regional coordinate vectors
    n_rows, n_cols : global grid dimensions

    Returns
    -------
    embedded_list : list of ndarray  (time, n_rows, n_cols)
    y_min, y_max, x_min, x_max : bounding-box indices into the global grid
    """
    n_time = regional_arrays[0].shape[0]

    embed_mask = (
        (lat_global >= lat_regional.min()) & (lat_global <= lat_regional.max()) &
        (lon_global >= lon_regional.min()) & (lon_global <= lon_regional.max())
    )
    coords = np.argwhere(embed_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    embedded_list = []
    for arr in regional_arrays:
        buf = np.full((n_time, n_rows, n_cols), np.nan)
        buf[:, y_min:y_max + 1, x_min:x_max + 1] = arr
        embedded_list.append(buf)

    return embedded_list, y_min, y_max, x_min, x_max


def normalise_inputs(sla, sst, ugos, vgos, params):
    """
    Compute phase channels and normalise all inputs.

    Returns
    -------
    x_input : ndarray  (time, rows, cols, 6)
        Model-ready input tensor (NaN replaced by 0).
    """
    phase1 = np.degrees(np.arctan2(vgos, ugos))
    phase2 = np.mod(phase1, 360)

    sst    = sst  - params["sst_mean"]
    sla    = sla  / params["ssh_maxabs"]
    sst    = sst  / params["sst_maxabs"]
    ugos   = ugos / params["ugosa_maxabs"]
    vgos   = vgos / params["vgosa_maxabs"]
    phase1 = phase1 / params["phase1_maxabs"]
    phase2 = phase2 / params["phase2_maxabs"]

    return np.nan_to_num(
        np.stack((sla, sst, ugos, vgos, phase1, phase2), axis=-1)
    )


# =============================================================================
# Post-processing: segmentation
# =============================================================================

def create_segmentation(pred_mask, logit=0.0, ocean_mask=None):
    """
    Binarise a raw probability map at the given logit threshold.

    Parameters
    ----------
    pred_mask : ndarray
    logit : float — binarisation threshold
    ocean_mask : ndarray or None — if provided, multiplied into the result
                 (land pixels become 0)
    """
    binary = np.where(pred_mask > logit, 1, 0)
    if ocean_mask is not None:
        binary = binary * ocean_mask
    return binary


# =============================================================================
# Geometry helpers
# =============================================================================

def haversine(lo1, lo2, la1, la2, degrees=True, latdep_rad=True):
    """Great-circle distance between two points (or arrays of points) in km."""
    lo1, lo2 = np.asanyarray(lo1), np.asanyarray(lo2)
    la1, la2 = np.asanyarray(la1), np.asanyarray(la2)

    if degrees:
        lo1, lo2, la1, la2 = map(np.radians, [lo1, lo2, la1, la2])

    if latdep_rad:
        a_e, b_e = 6378.1370, 6356.7523
        la0 = (la1 + la2) / 2
        numerator   = (a_e**2 * np.cos(la0))**2 + (b_e**2 * np.sin(la0))**2
        denominator = (a_e   * np.cos(la0))**2 + (b_e   * np.sin(la0))**2
        R = np.sqrt(numerator / denominator)
    else:
        R = 6371.0

    sin_dlat = np.sin((la2 - la1) / 2)
    sin_dlon = np.sin((lo2 - lo1) / 2)
    a = sin_dlat**2 + np.cos(la1) * np.cos(la2) * sin_dlon**2
    c = 2 * np.arctan2(np.sqrt(np.abs(a)), np.sqrt(1 - np.abs(a)))
    return R * c


def compute_pixel_areas(lat_grid, lon_grid):
    """Area (km²) of each pixel in a regular lat/lon grid."""
    dlat = np.abs(np.diff(lat_grid[:, 0])).mean()
    dlon = np.abs(np.diff(lon_grid[0, :])).mean()
    heights = haversine(lon_grid, lon_grid, lat_grid, lat_grid + dlat)
    widths  = haversine(lon_grid, lon_grid + dlon, lat_grid, lat_grid)
    return heights * widths


# =============================================================================
# Post-processing: active / inactive classification
# =============================================================================

def classify_eddies(regional_segmented, regional_sst_anomaly,
                    lat_grid, lon_grid,
                    area_threshold=300, sst_threshold=0.1):
    """
    Classify segmented eddies as active or inactive.

    Active   : area >= area_threshold  AND  max|SST anomaly| >= sst_threshold
    Inactive : area >= area_threshold  AND  max|SST anomaly| <  sst_threshold
    Discarded: area <  area_threshold

    Parameters
    ----------
    regional_segmented   : ndarray (time, lat, lon) — binary eddy mask
    regional_sst_anomaly : ndarray (time, lat, lon) — SST anomaly in °C
    lat_grid, lon_grid   : 2-D ndarray — regional coordinate grids
    area_threshold       : float — minimum eddy area in km²
    sst_threshold        : float — minimum peak |SST anomaly| in °C

    Returns
    -------
    active_eddies, inactive_eddies : ndarray (time, lat, lon), dtype uint8
    """
    active_eddies   = np.zeros_like(regional_segmented, dtype=np.uint8)
    inactive_eddies = np.zeros_like(regional_segmented, dtype=np.uint8)

    pixel_areas      = compute_pixel_areas(lat_grid, lon_grid)
    pixel_areas_flat = pixel_areas.ravel()
    structure        = np.ones((3, 3))  # 8-connectivity

    for day in range(regional_segmented.shape[0]):
        labeled, n_feat = label(regional_segmented[day], structure)
        if n_feat == 0:
            continue

        labels_flat = labeled.ravel()

        areas = np.bincount(
            labels_flat, weights=pixel_areas_flat, minlength=n_feat + 1
        )
        abs_sst = np.abs(regional_sst_anomaly[day])
        max_sst = nd_maximum(abs_sst, labels=labeled,
                             index=np.arange(n_feat + 1))

        is_large  = areas   >= area_threshold
        is_strong = max_sst >= sst_threshold

        active_mask            = is_large &   is_strong
        inactive_mask          = is_large & (~is_strong)
        active_mask[0]         = False
        inactive_mask[0]       = False

        active_eddies[day]   = np.where(active_mask[labeled],   1, 0)
        inactive_eddies[day] = np.where(inactive_mask[labeled], 1, 0)

    return active_eddies, inactive_eddies


# =============================================================================
# Output
# =============================================================================

def save_npy_outputs(output_dir, output_stem, detections, t_anom_preds):
    """
    Save raw model outputs as .npy files next to the input NetCDF.

    Parameters
    ----------
    output_dir  : str — directory where the input NetCDF lives
    output_stem : str — filename stem (no extension), derived from the input filename
    """
    np.save(os.path.join(output_dir, output_stem + "_detections.npy"),
            detections[..., None])
    np.save(os.path.join(output_dir, output_stem + "_t_anom_predictions.npy"),
            t_anom_preds[..., None])
    print(f"Saved .npy prediction arrays to {output_dir}/")


def append_to_netcdf(vardyn_file, active_eddies, inactive_eddies, regional_sst_anomaly):
    """
    Append eddy classification and SST anomaly as new variables directly into
    the input VarDyn NetCDF file. The file must already have
    (time, latitude, longitude) dimensions.

    Parameters
    ----------
    vardyn_file : str — full path to the VarDyn NetCDF (same as VARDYN_FILE)
    """
    with Dataset(vardyn_file, "a") as ds:
        v_active   = ds.createVariable("active_eddy",        "f4", ("time", "latitude", "longitude"))
        v_inactive = ds.createVariable("inactive_eddy",      "f4", ("time", "latitude", "longitude"))
        v_sst      = ds.createVariable("spatial_sst_anomaly","f4", ("time", "latitude", "longitude"))

        v_active[:]   = active_eddies
        v_inactive[:] = inactive_eddies
        v_sst[:]      = regional_sst_anomaly

        v_active.long_name   = ("Segmented thermodynamically active eddy "
                                "prediction (0=Ocean, 1=Eddy)")
        v_inactive.long_name = ("Segmented thermodynamically inactive eddy "
                                "prediction (0=Ocean, 1=Eddy)")
        v_sst.long_name      = ("Spatial SST anomaly reconstruction from "
                                "the model (°C)")

    print(f"Appended eddy variables to {vardyn_file}")
