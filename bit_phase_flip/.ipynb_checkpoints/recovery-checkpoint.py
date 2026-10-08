def recover(qc, state, syn):

    creg = qc.cregs[0]
    if state in ["0", "1"]:
       with qc.if_test((syn,0b10)):
          qc.x(2)

       with qc.if_test((syn,0b11)):
          qc.x(1)

       with qc.if_test((syn,0b01)):
          qc.x(0)
    else:
       with qc.if_test((syn,0b10)):
          qc.z(2)

       with qc.if_test((syn,0b11)):
          qc.z(1)

       with qc.if_test((syn,0b01)):
          qc.z(0)
    return qc