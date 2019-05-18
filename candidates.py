"""Collection of the current best candidates for nmrtools functions,
for testing speed etc.

"""


from nmrtools.nmrmath import hamiltonian as current_hamiltonian
from nmrtools.nmrmath import simsignals as current_simsignals

from speedtest.compare_hamiltonians import hamiltonian_sparse as \
    candidate_hamiltonian
from tests.test_simsignals import vectorized_simsignals as candidate_simsignals


if __name__ == '__main__':
    import numpy as np
    from simulation_data import rioux
    current_h = current_hamiltonian(*rioux())
    current_spectrum = current_simsignals(current_h, 3)
    candidate_h = candidate_hamiltonian(*rioux())
    candidate_spectrum = candidate_simsignals(candidate_h, 3)

    print(current_spectrum[:10])
    print(candidate_spectrum[:10])
    assert np.allclose(current_spectrum, candidate_spectrum)
