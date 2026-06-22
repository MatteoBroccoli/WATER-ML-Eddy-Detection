# WATER — VarDyn Eddy Detection (Field Campaign Branch)

This branch is **fully self-contained**: it requires no files from the main
branch and no external library beyond the packages in `requirements.txt`.

> **For the ship operator:** all you need to do each day is update one line in
> `namelist.py` and run one command. See [Daily workflow](#daily-workflow).

---

## Repository structure

```
vardyn_detection/
├── namelist.py          ← EDIT THIS before each run (paths & settings)
├── run_detection.py     ← main script — do not edit
├── detection_utils.py   ← frozen model, losses and helper functions — do not edit
└── requirements.txt
```

---

## One-time setup

### 1. Clone this branch

```bash
git clone --branch vardyn-detection \
    https://github.com/MatteoBroccoli/WATER-ML-Eddy-Detection.git
cd WATER-ML-Eddy-Detection
```

### 2. Create the conda environment

The environment is pinned to the same library versions used during training.

```bash
conda create -n water_det python=3.10 -y
conda activate water_det

# Install CUDA-aware TensorFlow first (adjust cudatoolkit to match the
# GPU drivers on the ship's machine)
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6 -y

# Install all Python dependencies
pip install -r requirements.txt
```

> **CPU-only machine?** Replace the `conda install cudatoolkit` step with:
> ```bash
> pip install tensorflow-cpu==2.13.0
> ```
> and remove `tensorflow==2.13.0` from `requirements.txt` before running
> `pip install -r requirements.txt`.

Verify the installation:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
# Expected: 2.13.0
```

### 3. Download model weights, mesh and normalisation parameters

All three files are archived on Zenodo.

> **TODO:** Replace the placeholder URL below with the actual Zenodo DOI/URL
> once the upload is complete.

```bash
# Create the model directory
mkdir -p /path/to/your/model_dir/cp/

# Download and unzip the model checkpoint
wget -O model_weights.zip https://zenodo.org/record/XXXXXXX/files/swot_tr014_checkpoint.zip
unzip model_weights.zip -d /path/to/your/model_dir/cp/

# Download the normalisation parameters
wget -O /path/to/your/model_dir/norm_params.npz \
    https://zenodo.org/record/XXXXXXX/files/norm_params.npz

# Download the mesh mask
wget -O /path/to/your/MESH/meshmask.nc \
    https://zenodo.org/record/XXXXXXX/files/meshmask.nc
```

The unzipped checkpoint folder should contain:
```
cp/
├── checkpoint
├── cp.index
└── cp.data-00000-of-00001
```

### 4. Edit `namelist.py`

Open `namelist.py` and update **all path variables** to match your machine:

```python
MESH_DIR         = "/path/to/MESH/"
MODEL_DIR        = "/path/to/model/"
NORM_PARAMS_PATH = MODEL_DIR + "norm_params.npz"
CHECKPOINT_DIR   = MODEL_DIR + "cp/"
```

Run a test with an example file to confirm everything works before the
campaign starts.

---

## Daily workflow

Each morning, before running detection:

1. **Update one line in `namelist.py`:**

```python
VARDYN_FILE = "/path/to/VarDyn_Agulhas_YYYYMMDD.nc"   # today's file
```

2. **Activate the environment and run:**

```bash
conda activate water_det
python run_detection.py
```

3. **Check the outputs** — everything is written into or alongside the input file:

| File | Description |
|---|---|
| `VarDyn_Agulhas_YYYYMMDD_detections.npy` | Raw eddy probability map |
| `VarDyn_Agulhas_YYYYMMDD_t_anom_predictions.npy` | Raw SST anomaly reconstruction |
| `VarDyn_Agulhas_YYYYMMDD.nc` | Input file with three new variables appended (see below) |

The new NetCDF variables appended to the input file are:

| Variable | Description |
|---|---|
| `active_eddy` | Binary mask — eddies with area ≥ 300 km² **and** peak \|SST\| ≥ 0.1 °C |
| `inactive_eddy` | Binary mask — eddies with area ≥ 300 km² **and** peak \|SST\| < 0.1 °C |
| `spatial_sst_anomaly` | SST anomaly reconstruction from the model (°C) |

---

## Detection settings

All tunable parameters are in `namelist.py`. The defaults below are the
values optimised for the SWOT-fine-tuned model — **do not change them
unless you have re-run the threshold optimisation**:

| Parameter | Default | Meaning |
|---|---|---|
| `LOGIT_THRESHOLD` | 0.208 | Binarisation threshold for raw eddy probabilities |
| `SST_THRESHOLD` | 0.1 °C | Minimum peak \|SST anomaly\| for an eddy to be "active" |
| `AREA_THRESHOLD` | 300 km² | Minimum eddy size (smaller eddies are discarded) |

---

## Troubleshooting

**`FileNotFoundError: No checkpoint found`**  
The model weights were not found in `CHECKPOINT_DIR`. Check the path in
`namelist.py` and re-run the download step above.

**Out-of-memory on GPU**  
Pass `batch_size=1` to `model.predict()` in `run_detection.py`
(line: `predictions = model.predict(x_input, ...)`).

**`variable already exists` error when appending to NetCDF**  
The three output variables already exist in the file from a previous run.
Either delete them first or use a fresh copy of the input file.

---

## Contact

Matteo Broccoli — CMCC  
`mb31322 [at] cmcc [dot] it`
