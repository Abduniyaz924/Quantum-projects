def prepare_state(qc, state):
    if state == "0":
        pass
    elif state == "1" or "-":
        qc.x(0)
    qc.cx(0,1)
    qc.cx(0,2)
    if state in ["+", "-"]:
        qc.h(0)
        qc.h(1)
        qc.h(2)
    return qc