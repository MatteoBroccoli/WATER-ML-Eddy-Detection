# =============================================================================
# NAMELIST — VarDyn Eddy Detection
# =============================================================================
# Edit ONLY this file before running the detection.
# All paths and parameters that may differ across machines are set here.
# =============================================================================

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

# Directory containing the mesh file (meshmask.nc)
MESH_DIR = "/work/cmcc/mb31322/water/DET/AUGUST_2025/MESH/"

# Path to the trained model directory (contains norm_params.npz and cp/)
MODEL_DIR = "/work/cmcc/mb31322/water/unet/swot/tr014/"

# Path to the normalisation parameters file
NORM_PARAMS_PATH = MODEL_DIR + "norm_params.npz"

# Path to the model checkpoint subfolder
CHECKPOINT_DIR = MODEL_DIR + "cp/"

# -----------------------------------------------------------------------------
# INPUT FILE
# -----------------------------------------------------------------------------

# Full path to the daily VarDyn input NetCDF file.
# Update this every day before running — the detections will be appended here.
VARDYN_FILE = "/work/cmcc/mb31322/water/vardyn/VarDyn_Agulhas_20240601.nc"

# -----------------------------------------------------------------------------
# DETECTION SETTINGS
# -----------------------------------------------------------------------------

# Logit threshold for binarising the raw eddy probability map.
# Optimised value from eval/; adjust only if you re-run threshold optimisation.
LOGIT_THRESHOLD = 0.208

# Minimum peak |SST anomaly| (°C) inside an eddy to be considered "active".
SST_THRESHOLD = 0.1

# Minimum eddy area (km²) — eddies smaller than this are discarded entirely.
AREA_THRESHOLD = 300
