from simulation_data import spin11, rioux
from candidates import candidate_hamiltonian, candidate_simsignals
from tests.test_simsignals import H_RIOUX, H11_NDARRAY

@profile
def test_h_mem():
    return candidate_hamiltonian(*rioux())


@profile
def test_spec_mem(v, J):
    h = candidate_hamiltonian(v, J)
    return candidate_simsignals(h, len(v))


if __name__ == '__main__':
    test_spec_mem(*spin11())
