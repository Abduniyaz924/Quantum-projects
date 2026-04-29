from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator

estimator = StatevectorEstimator()

ansatz = TwoLocal(
    rotation_blocks='ry',
    entanglement_blocks='cz',
    reps=2
)

optimizer = COBYLA()

vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer
)

#result = vqe.compute_minimum_eigenvalue(H)

#print(result.eigenvalue)
#print(result.optimal_point)
