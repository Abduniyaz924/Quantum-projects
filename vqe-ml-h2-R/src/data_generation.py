import numpy as np
from hamiltonian import generate_hamiltonian
from vqe import vqe
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

os.makedirs(DATA_DIR, exist_ok=True)  # create if missing

SAVE_PATH = os.path.join(DATA_DIR, "vqe_dataset.csv")

bond_lengths = np.linspace(0.4, 2.0, 100)

dataset = []

for R in bond_lengths:

    H = generate_hamiltonian(R)
    print("R is :", R)

    result = vqe.compute_minimum_eigenvalue(H)

    data_point = {
        "R": R,
        "energy": result.eigenvalue.real,
        "params": result.optimal_point
    }

    dataset.append(data_point)


import pandas as pd

rows = []

for d in dataset:
    row = {"R": d["R"], "energy": d["energy"]}
    
    for i,p in enumerate(d["params"]):
        row[f"theta_{i}"] = p
    
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(SAVE_PATH)
