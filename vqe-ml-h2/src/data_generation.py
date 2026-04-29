import os
import numpy as np
from scipy.optimize import minimize
from vqe import cost_function

# number of samples
NUM_SAMPLES = 100

# -------------------------
# Paths (robust)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

os.makedirs(DATA_DIR, exist_ok=True)  # create if missing

SAVE_PATH = os.path.join(DATA_DIR, "vqe_dataset.npy")

# -------------------------
# Generate data
# -------------------------
data = []

for i in range(NUM_SAMPLES):
    print(f"Running sample {i+1}/{NUM_SAMPLES}")
    
    theta_init = np.random.uniform(0, 2*np.pi, 4)
    
    result = minimize(cost_function, theta_init, method="COBYLA")
    
    data.append({
        "theta_init": theta_init,
        "theta_opt": result.x,
        "energy": result.fun
    })

# -------------------------
# Save dataset
# -------------------------
np.save(SAVE_PATH, data)

print(f"Dataset saved at: {SAVE_PATH}")
