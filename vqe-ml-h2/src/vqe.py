import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------
# 1. Define Hamiltonian (H2 simplified)
# -----------------------------
hamiltonian = SparsePauliOp.from_list([
    ("II", -1.052373),
    ("IZ", 0.397937),
    ("ZI", -0.397937),
    ("ZZ", -0.011280),
    ("XX", 0.180931)
])

# -----------------------------
# 2. Ansatz (parameterized circuit)
# -----------------------------
def ansatz(theta):
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.ry(theta[2], 0)
    qc.ry(theta[3], 1)
    return qc

# -----------------------------
# 3. Cost function (Energy)
# -----------------------------
# Use StatevectorEstimator in Qiskit 2.3.0
estimator = StatevectorEstimator()

def cost_function(theta):
    qc = ansatz(theta)

    job = estimator.run([(qc, hamiltonian)])
    result = job.result()

    # Qiskit 2.x result extraction
    energy = result[0].data.evs

    return float(np.real(energy))



# -----------------------------
# 4. Optimization
# -----------------------------
initial_theta = np.random.rand(4)
energies = []

def callback(xk):
    energy = cost_function(xk)
    energies.append(energy)

result = minimize(
    cost_function,
    initial_theta,
    method="COBYLA",
    callback=callback
)

# -----------------------------
# 5. Results
# -----------------------------
print("Final Energy:", result.fun)
# Initial circuit
initial_qc = ansatz(initial_theta)
initial_qc.draw("mpl")

# Final circuit
optimal_qc = ansatz(result.x)
optimal_qc.draw("mpl")
optimal_qc.decompose().draw("mpl")
