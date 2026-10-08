def measure_Z_stabilizer(qc, data, anc, syn, stabilizer, ancilla_index, syn_index):

    for q in stabilizer:
        qc.cx(data[q], anc[ancilla_index])

    qc.measure(anc[ancilla_index], syn[syn_index])


def measure_X_stabilizer(qc, data, anc, syn, stabilizer, ancilla_index, syn_index):

    qc.h(anc[ancilla_index])

    for q in stabilizer:
        qc.cx(anc[ancilla_index], data[q])

    qc.h(anc[ancilla_index])

    qc.measure(anc[ancilla_index], syn[syn_index])