def encode_shor(qc):

    # Stage 1: create three outer-code qubits
    qc.cx(0, 3)
    qc.cx(0, 6)

    # Stage 2: put the three outer qubits into X basis
    qc.h(0)
    qc.h(3)
    qc.h(6)

    # Stage 3: expand each outer qubit into a 3-qubit repetition block

    # Block 1
    qc.cx(0, 1)
    qc.cx(0, 2)

    # Block 2
    qc.cx(3, 4)
    qc.cx(3, 5)

    # Block 3
    qc.cx(6, 7)
    qc.cx(6, 8)

    return qc