import os, sys
import datetime as dt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Path setup to find src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import get_unet
from src.losses import combined_eddy_loss, detection_metric, temp_metric, boundary_sst_coupling_loss, gradient_loss
from src import data_utils as du

# --- CONFIGURATION ---
DATA_ROOT = '/work/cmcc/mb31322/water/DET/AUGUST_2025/'
OUTPUT_DIR = '/work/cmcc/mb31322/water/unet/curriculum_results/'
BATCH_SIZE = 4

# Phase 1: Nadir Pre-training
PHASE1_DATES = (dt.datetime(2022, 1, 2), dt.datetime(2023, 7, 26))
PHASE1_LR = 1e-4
PHASE1_EPOCHS = 500

# Phase 2: SWOT Fine-tuning
PHASE2_DATES = (dt.datetime(2023, 7, 27), dt.datetime(2024, 5, 1))
PHASE2_LR = 1e-5
PHASE2_EPOCHS = 250

def prepare_phase_data(start_date, end_date, data_type, stats=None):
    """Loads and normalizes data. If stats are provided, use them for scaling."""
    print(f"Preparing {data_type} data from {start_date.date()} to {end_date.date()}...")
    ssh, sst, w_act, w_exc, t_act, t_exc, u, v = du.load_data(start_date, end_date, DATA_ROOT, data_type)
    p1, p2 = du.compute_phase(u, v)
    
    # Static Land Mask
    mask = np.where(ssh > 8e+36, 0, 1)
    
    # Compute SST Anomaly (Simplified - usually done against a climatology)
    sst_mean = np.nanmean(sst, axis=0)
    sst_anom = sst - sst_mean

    # Target: Combined Eddy Mask and Normalized Temp Anomaly
    y_mask = w_act + w_exc
    y_temp = (t_act + t_exc)

    # Scaling Statistics
    if stats is None:
        stats = {
            'ssh': np.nanmax(np.abs(ssh)),
            'sst': np.nanmax(np.abs(sst_anom)),
            'u': np.nanmax(np.abs(u)),
            'v': np.nanmax(np.abs(v)),
            't_target': np.nanmax(np.abs(y_temp))
        }
        print("Generated new scaling stats from this dataset.")

    # Normalization
    x = np.nan_to_num(np.stack((
        ssh / stats['ssh'], 
        sst_anom / stats['sst'], 
        u / stats['u'], 
        v / stats['v'], 
        p1 / 180.0, 
        p2 / 360.0
    ), axis=-1))
    
    y = np.nan_to_num(np.stack((y_mask, y_temp / stats['t_target']), axis=-1))
    
    # Masked Weights for Loss
    weights = np.where(y_mask == 1, 2.0, 0.67) * mask

    return x, y, weights, stats

def run_curriculum():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- PHASE 1: NADIR PRE-TRAINING ---
    x1, y1, w1, nadir_stats = prepare_phase_data(*PHASE1_DATES, 'NADIR')
    x1_tr, x1_val = du.split_train_validation(x1)
    y1_tr, y1_val = du.split_train_validation(y1)
    w1_tr, w1_val = du.split_train_validation(w1)

    train_gen1 = du.MaskedGenerator(x1_tr, y1_tr, w1_tr, BATCH_SIZE, shuffle=True)
    val_gen1 = du.MaskedGenerator(x1_val, y1_val, w1_val, BATCH_SIZE)

    model = get_unet(input_shape=(1701, 3600, 6), output_channels=2, num_stages=3)
    model.compile(optimizer=keras.optimizers.Adam(PHASE1_LR), loss=combined_eddy_loss,
                  weighted_metrics=[detection_metric, temp_metric, boundary_sst_coupling_loss, gradient_loss])

    print("\nStarting Phase 1: Nadir Pre-training...")
    cp_p1 = os.path.join(OUTPUT_DIR, "phase1_nadir/cp/")
    callbacks1 = [
        du.BestCheckpointWithEpoch(filepath=cp_p1, monitor="val_loss"),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15),
        keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "history_phase1.csv"))
    ]
    model.fit(train_gen1, validation_data=val_gen1, epochs=PHASE1_EPOCHS, callbacks=callbacks1, verbose=2)

    # --- TRANSITION ---
    print("\nTransitioning to Phase 2. Loading best Nadir weights...")
    best_p1 = tf.train.latest_checkpoint(cp_p1)
    model.load_weights(best_p1)

    # --- PHASE 2: SWOT FINE-TUNING ---
    # We pass nadir_stats to ensure feature consistency during transfer
    x2, y2, w2, _ = prepare_phase_data(*PHASE2_DATES, 'SWOT', stats=nadir_stats)
    x2_tr, x2_val = du.split_train_validation(x2)
    y2_tr, y2_val = du.split_train_validation(y2)
    w2_tr, w2_val = du.split_train_validation(w2)

    train_gen2 = du.MaskedGenerator(x2_tr, y2_tr, w2_tr, BATCH_SIZE, shuffle=True)
    val_gen2 = du.MaskedGenerator(x2_val, y2_val, w2_val, BATCH_SIZE)

    # Re-compile with lower learning rate for fine-tuning
    model.compile(optimizer=keras.optimizers.Adam(PHASE2_LR), loss=combined_eddy_loss,
                  weighted_metrics=[detection_metric, temp_metric, boundary_sst_coupling_loss, gradient_loss])

    print("\nStarting Phase 2: SWOT Fine-tuning...")
    cp_p2 = os.path.join(OUTPUT_DIR, "phase2_swot/cp/")
    callbacks2 = [
        du.BestCheckpointWithEpoch(filepath=cp_p2, monitor="val_loss"),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15),
        keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "history_phase2.csv"))
    ]
    model.fit(train_gen2, validation_data=val_gen2, epochs=PHASE2_EPOCHS, callbacks=callbacks2, verbose=2)

    # --- FINAL PREDICTIONS ---
    print("\nTraining Complete. Generating final validation predictions...")
    best_p2 = tf.train.latest_checkpoint(cp_p2)
    model.load_weights(best_p2)
    
    # Predict on SWOT validation set
    preds = model.predict(x2_val, batch_size=BATCH_SIZE, verbose=1)
    
    # Save results for evaluation scripts
    np.save(os.path.join(OUTPUT_DIR, 'val_preds_mask.npy'), preds[..., 0])
    # Denormalize Temp Preds using the Nadir stats preserved earlier
    t_preds = preds[..., 1] * nadir_stats['t_target']
    np.save(os.path.join(OUTPUT_DIR, 'val_preds_temp.npy'), t_preds)
    
    print(f"Results and history saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_curriculum()