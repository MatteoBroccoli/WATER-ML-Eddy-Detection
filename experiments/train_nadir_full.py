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

# Configuration
DATA_ROOT = '/work/cmcc/mb31322/water/DET/AUGUST_2025/'
TRAIN_DIR = '/work/cmcc/mb31322/water/unet/nadir/tr067/'
BATCH_SIZE = 4
EPOCHS = 500
LR = 1e-4
CLASS_WEIGHTS = [0.67, 2.00]

def run_training():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    start_date = dt.datetime(2022, 1, 2)
    end_date = dt.datetime(2024, 5, 1)

    # Load Data
    print("Loading Nadir Data...")
    ssh, sst, w_act, w_exc, t_act, t_exc, u, v = du.load_data(start_date, end_date, DATA_ROOT, 'NADIR')
    p1, p2 = du.compute_phase(u, v)
    
    # Masking and Prep
    mask = np.where(ssh > 8e+36, 0, 1)
    ssh = np.where(mask == 1, ssh, np.nan)
    sst = np.where(mask == 1, sst, np.nan)
    u = np.where(mask == 1, u, np.nan)
    v = np.where(mask == 1, v, np.nan)
    t_act = np.where(mask == 1, t_act, np.nan)
    t_exc = np.where(mask == 1, t_exc, np.nan)
    t = t_act + t_exc
    
    # Split
    ssh_tr, ssh_val = du.split_train_validation(ssh)
    sst_tr, sst_val = du.split_train_validation(sst)
    u_tr, u_val = du.split_train_validation(u); v_tr, v_val = du.split_train_validation(v)
    p1_tr, p1_val = du.split_train_validation(p1); p2_tr, p2_val = du.split_train_validation(p2)
    t_tr, _ = du.split_train_validation(t)
    
    sst_tr, sst_val, _ = du.anomaly(ssh_tr, sst_val, sst_tr) # Compute anomaly before normalization
    
    # Normalize (Using maxabs logic from original script)
    ssh_max = np.nanmax(np.abs(ssh_tr))
    sst_max = np.nanmax(np.abs(sst_tr))
    u_max = np.nanmax(np.abs(u_tr)); v_max = np.nanmax(np.abs(v_tr))
    p1_max = 180.0; p2_max = 360.0
    t_max = np.nanmax(np.abs(t_tr))
    
    x_tr = np.nan_to_num(np.stack((ssh_tr/ssh_max, sst_tr/sst_max, u_tr/u_max, v_tr/v_max, p1_tr/p1_max, p2_tr/p2_max), axis=-1))
    x_val = np.nan_to_num(np.stack((ssh_val/ssh_max, sst_val/sst_max, u_val/u_max, v_val/v_max, p1_val/p1_max, p2_val/p2_max), axis=-1))
    
    y_mask = du.split_train_validation(w_act + w_exc)
    y_temp = du.split_train_validation((t_act + t_exc) / t_max) # Normalized by 1.5 approx maxabs
    
    y_tr = np.nan_to_num(np.stack((y_mask[0], y_temp[0]), axis=-1))
    y_val = np.nan_to_num(np.stack((y_mask[1], y_temp[1]), axis=-1))

    # Generators
    m_tr, m_val = du.split_train_validation(mask)
    w_tr = np.where(y_mask[0] == 1, CLASS_WEIGHTS[1], CLASS_WEIGHTS[0]) * m_tr
    w_val = np.where(y_mask[1] == 1, CLASS_WEIGHTS[1], CLASS_WEIGHTS[0]) * m_val

    train_gen = du.MaskedGenerator(x_tr, y_tr, w_tr, BATCH_SIZE, shuffle=True)
    val_gen = du.MaskedGenerator(x_val, y_val, w_val, BATCH_SIZE)

    # Model
    model = get_unet(input_shape=(1701, 3600, 6), output_channels=2, num_stages=3)
    model.compile(optimizer=keras.optimizers.Adam(LR), loss=combined_eddy_loss, 
                  weighted_metrics=[detection_metric, temp_metric, boundary_sst_coupling_loss, gradient_loss])

    # Callbacks
    cp_path = os.path.join(TRAIN_DIR, "cp/")
    callbacks = [
        du.BestCheckpointWithEpoch(filepath=cp_path, monitor="val_loss"),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
        keras.callbacks.CSVLogger(os.path.join(TRAIN_DIR, "training.csv"))
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks, verbose=2)

if __name__ == "__main__":
    run_training()