import numpy as np
# The best Kuprov algorithm so far
from kuprov import kuprov_H_dense as kuprov

# My original slower Hamilton
from nmrtools.nmrmath import hamiltonian_slow

# My faster hamiltonian using pre-saved partial solutions/sparse matrices
from nmrtools.nmrmath import hamiltonian

# My new vectorized Hamiltonian
# ref: https://stackoverflow.com/questions/47752324/matrix-multiplication-on-4d-numpy-arrays
# Part 1: pre-compute L operators


def spin_operators(nspins):
    sigma_x = np.array([[0, 1 / 2], [1 / 2, 0]])
    sigma_y = np.array([[0, -1j / 2], [1j / 2, 0]])
    sigma_z = np.array([[1 / 2, 0], [0, -1 / 2]])
    unit = np.array([[1, 0], [0, 1]])

    L = np.empty((3, nspins, 2 ** nspins, 2 ** nspins), dtype=np.complex128)  # consider other dtype?
    for n in range(nspins):
        Lx_current = 1
        Ly_current = 1
        Lz_current = 1

        for k in range(nspins):
            if k == n:
                Lx_current = np.kron(Lx_current, sigma_x)
                Ly_current = np.kron(Ly_current, sigma_y)
                Lz_current = np.kron(Lz_current, sigma_z)
            else:
                Lx_current = np.kron(Lx_current, unit)
                Ly_current = np.kron(Ly_current, unit)
                Lz_current = np.kron(Lz_current, unit)

        L[0][n] = Lx_current
        L[1][n] = Ly_current
        L[2][n] = Lz_current

    return L


# Part 2: vectorized Hamiltonian using spin_operators
# First: an unvectorized Hamiltonian using spin_operators


def hamiltonian_unvectorized(v, J, L):
    nspins = len(v)
    Lx = L[0]
    Ly = L[1]
    Lz = L[2]
    H = np.tensordot(v, L[2], axes=1)

    for n in range(nspins):
        for k in range(nspins):
            if n != k:
                H += 0.5 * J[n, k] * (Lx[n] @ Lx[k] + Ly[n] @ Ly[k] + Lz[n] @ Lz[k])

    return H


# Then: vectorized
def hamiltonian_vectorized(v, J, L):
    nspins = len(v)
    H = np.tensordot(v, L[2], axes=1)
    L_T = L.transpose(1, 0, 2, 3)
    Lproduct = np.tensordot(L_T, L, axes=((1, 3), (0, 2))).swapaxes(1, 2)
    scalars = 0.5 * J
    H += np.tensordot(scalars, Lproduct, axes=2)
    return H


if __name__ == '__main__':
    from simulation_data import spin8
    v, J = spin8()
    L = spin_operators(8)
    hamiltonians = [kuprov(v, J), hamiltonian_slow(v, J), hamiltonian(v, J), hamiltonian_unvectorized(v, J, L),
                    hamiltonian_vectorized(v, J, L)]
    for i in range(len(hamiltonians)-1):
        assert np.array_equal(hamiltonians[i], hamiltonians[i+1])
