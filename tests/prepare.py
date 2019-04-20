"""Hacking in references for testing"""

from nmrtools.nmrmath import hamiltonian_slow
from simulation_data import spin8


v, J = spin8()
standard_H = hamiltonian_slow(v, J)
