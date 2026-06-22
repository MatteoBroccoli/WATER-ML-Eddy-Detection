"""
run_detection.py
================
Daily eddy detection on VarDyn data using the SWOT-fine-tuned U-Net.

Usage
-----
    python run_detection.py

Before running, open namelist.py and set VARDYN_FILE to today's input file.
The three output variables (active_eddy, inactive_eddy, spatial_sst_anomaly)
will be appended directly to that same NetCDF file. Raw model outputs (.npy)
are saved alongside the input file.
"""

import os
import gc
import numpy as np
import tensorflow as tf

import namelist as cfg
from detection_utils import (
    # model & losses
    get_unet,
    combined_eddy_loss,
    detection_metric,
    temp_metric,
    boundary_sst_coupling_loss,
    gradient_loss,
    # data pipeline
    load_mesh,
    load_norm_params,
    load_vardyn,
    mask_fill_values,
    kelvin_to_celsius,
    embed_regional_data,
    normalise_inputs,
    # post-processing
    create_segmentation,
    classify_eddies,
    # output
    save_npy_outputs,
    append_to_netcdf,
)


def main():
    tf.random.set_seed(cfg.TF_SEED)

    vardyn_dir  = os.path.dirname(cfg.VARDYN_FILE)
    output_stem = os.path.splitext(os.path.basename(cfg.VARDYN_FILE))[0]

    # ------------------------------------------------------------------
    # 1. Load mesh and normalisation parameters
    # ------------------------------------------------------------------
    print("Loading mesh and normalisation parameters...")
    _, lat_global, lon_global = load_mesh(cfg.MESH_DIR)
    norm = load_norm_params(cfg.NORM_PARAMS_PATH)

    # ------------------------------------------------------------------
    # 2. Load and pre-process VarDyn data
    # ------------------------------------------------------------------
    print(f"Loading VarDyn data from: {cfg.VARDYN_FILE}")
    sla, sst, ugos, vgos, lat_reg, lon_reg, _ = load_vardyn(cfg.VARDYN_FILE)

    sla, sst, ugos, vgos = mask_fill_values(sla, sst, ugos, vgos)
    ocean_mask = np.where(np.isnan(sla), np.nan, 1)
    sst = kelvin_to_celsius(sst)

    print("Embedding regional data into global grid...")
    (sla_emb, sst_emb, ugos_emb, vgos_emb), y_min, y_max, x_min, x_max = \
        embed_regional_data(
            [sla, sst, ugos, vgos],
            lat_global, lon_global,
            lat_reg, lon_reg,
            cfg.MODEL_INPUT_ROWS, cfg.MODEL_INPUT_COLS,
        )
    del sla, sst, ugos, vgos
    gc.collect()

    print("Normalising inputs...")
    x_input = normalise_inputs(sla_emb, sst_emb, ugos_emb, vgos_emb, norm)
    del sla_emb, sst_emb, ugos_emb, vgos_emb
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Build model and load weights
    # ------------------------------------------------------------------
    print("Building model and loading pre-trained weights...")
    model = get_unet(
        input_shape=(cfg.MODEL_INPUT_ROWS, cfg.MODEL_INPUT_COLS,
                     cfg.MODEL_INPUT_CHANNELS),
        output_channels=cfg.MODEL_OUTPUT_CHANNELS,
        num_stages=cfg.MODEL_NUM_STAGES,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=combined_eddy_loss,
        weighted_metrics=[
            detection_metric,
            temp_metric,
            boundary_sst_coupling_loss,
            gradient_loss,
        ],
    )

    latest = tf.train.latest_checkpoint(cfg.CHECKPOINT_DIR)
    if latest is None:
        raise FileNotFoundError(
            f"No checkpoint found in {cfg.CHECKPOINT_DIR}. "
            "Please download model weights — see README.md."
        )
    model.load_weights(latest).expect_partial()
    print(f"  Loaded: {latest}")

    # ------------------------------------------------------------------
    # 4. Run inference
    # ------------------------------------------------------------------
    print("Running inference...")
    predictions  = model.predict(x_input, verbose=1)
    detections   = predictions[..., 0]
    t_anom_preds = predictions[..., 1] * norm["t_anom_maxabs"]
    print(f"Predictions shape: {predictions.shape}")
    del x_input, predictions
    gc.collect()

    # ------------------------------------------------------------------
    # 5. Save raw .npy outputs
    # ------------------------------------------------------------------
    save_npy_outputs(vardyn_dir, output_stem, detections, t_anom_preds)

    # ------------------------------------------------------------------
    # 6. Post-process: segment and classify
    # ------------------------------------------------------------------
    print("Post-processing...")
    reg_det   = detections[:, y_min:y_max + 1, x_min:x_max + 1] * ocean_mask
    reg_t_ano = t_anom_preds[:, y_min:y_max + 1, x_min:x_max + 1] * ocean_mask
    del detections, t_anom_preds
    gc.collect()

    segmented    = create_segmentation(reg_det, logit=cfg.LOGIT_THRESHOLD)
    regional_lat = lat_global[y_min:y_max + 1, x_min:x_max + 1]
    regional_lon = lon_global[y_min:y_max + 1, x_min:x_max + 1]

    active, inactive = classify_eddies(
        segmented, reg_t_ano,
        regional_lat, regional_lon,
        area_threshold=cfg.AREA_THRESHOLD,
        sst_threshold=cfg.SST_THRESHOLD,
    )
    print(f"  Active eddies:   area >= {cfg.AREA_THRESHOLD} km2  AND  max|SST| >= {cfg.SST_THRESHOLD} degC")
    print(f"  Inactive eddies: area >= {cfg.AREA_THRESHOLD} km2  AND  max|SST| <  {cfg.SST_THRESHOLD} degC")

    # ------------------------------------------------------------------
    # 7. Append results to the input NetCDF file
    # ------------------------------------------------------------------
    append_to_netcdf(cfg.VARDYN_FILE, active, inactive, reg_t_ano)

    print("Done.")


if __name__ == "__main__":
    main()
