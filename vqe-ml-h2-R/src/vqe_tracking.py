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
