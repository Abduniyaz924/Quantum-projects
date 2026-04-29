import numpy as np
from hamiltonian import generate_hamiltonian
from vqe import vqe

bond_lengths = np.linspace(0.4, 2.0, 20)

dataset = []

for R in bond_lengths:

    H = generate_hamiltonian(R)

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
df.to_csv("vqe_dataset.csv", index=False)
