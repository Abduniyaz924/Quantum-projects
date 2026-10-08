from qiskit import QuantumCircuit

def encode(qc):
    qc.h(0)

    qc.cx(0,1)
    qc.cx(0,2)

    return qc