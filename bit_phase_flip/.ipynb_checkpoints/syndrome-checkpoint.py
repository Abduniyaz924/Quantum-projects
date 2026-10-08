def extract_syndrome(qc, state, syn):

    if state in ["0", "1"]: 
       # Z1Z2
       qc.cx(0,3)
       qc.cx(1,3)

       # Z2Z3
       qc.cx(1,4)
       qc.cx(2,4)
    else:
        # X0 X1
        qc.h(3)
        qc.cx(3, 0)
        qc.cx(3, 1)
        qc.h(3)

        # X1 X2
        qc.h(4)
        qc.cx(4, 1)
        qc.cx(4, 2)
        qc.h(4)

    qc.measure(3, syn[0])
    qc.measure(4, syn[1])

    return qc