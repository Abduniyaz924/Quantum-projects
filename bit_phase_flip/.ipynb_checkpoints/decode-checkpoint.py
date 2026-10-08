def decode(qc, state):

    if state in ["0", "1"]:
       qc.cx(0,2)
       qc.cx(0,1)

    else:

       # Transform X-basis code back to computational basis
        qc.h(0)
        qc.h(1)
        qc.h(2)

        qc.cx(0, 2)
        qc.cx(0, 1)

    return qc