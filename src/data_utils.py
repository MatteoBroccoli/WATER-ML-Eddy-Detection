import numpy as np
from netCDF4 import Dataset
from tensorflow import keras
import datetime as dt
import os
import glob

class MaskedGenerator(keras.utils.Sequence):
    def __init__(self, x, y, mask, batch_size, shuffle=False):
        self.x, self.y, self.mask = x, y, mask
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        return self.x[batch_indices], self.y[batch_indices], self.mask[batch_indices]

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

def load_data(start_date, end_date, base_path, data_type='NADIR'):
    """Helper to load all physical variables for a date range."""
    ssh, sst, mask_active, mask_exc, t_anom, t_exc = [], [], [], [], [], []
    u, v = [], []
    curr = start_date
    delta = dt.timedelta(days=1)
    
    while curr <= end_date:
        # Detected Eddies
        ds = Dataset(f"{base_path}/{data_type}/Detected_eddies_{curr.strftime('%Y%m%d')}.nc")
        ssh.append(ds.variables['SSH'][:]); sst.append(ds.variables['temp'][:])
        mask_active.append(ds.variables['Wtot'][:]); mask_exc.append(ds.variables['Wtot_exc'][:])
        t_anom.append(ds.variables['T_anom'][:]); t_exc.append(ds.variables['T_exc_anom'][:])
        ds.close()
        # Velocities
        uv_type = "NADIR_UV" if data_type == 'NADIR' else "SWOT_UV"
        ds_uv = Dataset(f"{base_path}/{uv_type}/{curr.strftime('%Y%m%d')}_SSH_DYN.nc")
        u.append(ds_uv.variables['ugosa'][:]); v.append(ds_uv.variables['vgosa'][:])
        ds_uv.close()
        curr += delta
        
    return (np.stack(ssh), np.stack(sst), np.stack(mask_active), np.stack(mask_exc), 
            np.stack(t_anom), np.stack(t_exc), np.stack(u), np.stack(v))

def compute_phase(u, v):
    p1 = np.degrees(np.arctan2(v, u))
    p2 = np.mod(p1, 360)
    return p1, p2

def split_train_validation(data, perc=0.8):
    split_index = int(len(data) * perc)
    return data[:split_index], data[split_index:]

def anomaly(train, test, reference):
    '''
    This compute the anomaly in the train and test sets, w.r.t. reference data.
    '''
    reference_mean = np.nanmean(reference, axis=0)
    train_anomaly = train - reference_mean
    test_anomaly = test - reference_mean
    return train_anomaly, test_anomaly, reference_mean