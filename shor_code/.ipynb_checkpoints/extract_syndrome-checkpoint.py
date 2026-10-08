from stabilizer import measure_Z_stabilizer, measure_X_stabilizer
def extract_syndrome(qc, data, anc, syn):

    # Z-type stabilizers
    measure_Z_stabilizer(qc, data, anc, syn, [0, 1], 0, 0)
    measure_Z_stabilizer(qc, data, anc, syn, [1, 2], 1, 1)

    measure_Z_stabilizer(qc, data, anc, syn, [3, 4], 2, 2)
    measure_Z_stabilizer(qc, data, anc, syn, [4, 5], 3, 3)

    measure_Z_stabilizer(qc, data, anc, syn, [6, 7], 4, 4)
    measure_Z_stabilizer(qc, data, anc, syn, [7, 8], 5, 5)

    # X-type stabilizers
    measure_X_stabilizer(
        qc, data, anc, syn,
        [0, 1, 2, 3, 4, 5], 6, 6
    )

    measure_X_stabilizer(
        qc, data, anc, syn,
        [3, 4, 5, 6, 7, 8], 7, 7
    )

    return qc


def extract_syndrome(qc, data, anc, x_syn, z_syn):

    # Z-type stabilizers
    measure_Z_stabilizer(qc, data, anc, x_syn, [0, 1], 0, 0)
    measure_Z_stabilizer(qc, data, anc, x_syn, [1, 2], 1, 1)

    measure_Z_stabilizer(qc, data, anc, x_syn, [3, 4], 2, 2)
    measure_Z_stabilizer(qc, data, anc, x_syn, [4, 5], 3, 3)

    measure_Z_stabilizer(qc, data, anc, x_syn, [6, 7], 4, 4)
    measure_Z_stabilizer(qc, data, anc, x_syn, [7, 8], 5, 5)

    # X-type stabilizers
    measure_X_stabilizer(
        qc, data, anc, z_syn,
        [0, 1, 2, 3, 4, 5], 6, 0
    )

    measure_X_stabilizer(
        qc, data, anc, z_syn,
        [3, 4, 5, 6, 7, 8], 7, 1
    )