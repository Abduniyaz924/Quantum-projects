from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

def generate_hamiltonian(bond_length):

    molecule = f"H 0 0 0; H 0 0 {bond_length}"

    driver = PySCFDriver(
        atom=molecule,
        basis="sto3g",
        charge=0,
        spin=0
    )

    problem = driver.run()

    second_q_op = problem.hamiltonian.second_q_op()

    mapper = JordanWignerMapper()
    qubit_hamiltonian = mapper.map(second_q_op)

    return qubit_hamiltonian
