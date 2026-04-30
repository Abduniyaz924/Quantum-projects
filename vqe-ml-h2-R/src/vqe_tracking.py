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

def run_vqe_with_tracking(H, initial_point):

    energies = []

    def callback(eval_count, params, value, metadata):
        energies.append(value)

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
        callback=callback
    )

    result = vqe.compute_minimum_eigenvalue(H)

    return energies, result
