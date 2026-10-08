def measure_state(qc, state, out):
    #if state in ["+", "-"]:
        #qc.h(0)

    qc.measure(0, out[0])
    return qc