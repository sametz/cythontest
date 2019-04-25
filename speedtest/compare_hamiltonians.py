import numpy as np
from scipy.sparse import csr_matrix
import sparse

from speedtest.speedutils import timefn  # Decorator for timing functions
# The best Kuprov algorithm so far
from speedtest.kuprov import kuprov_H_dense as kuprov_H
from speedtest.kuprov import kuprov_cached

# My original slower Hamilton
from nmrtools.nmrmath import hamiltonian_slow

# My faster hamiltonian using pre-saved partial solutions/sparse matrices
from nmrtools.nmrmath import hamiltonian

# My new vectorized Hamiltonian
# ref: https://stackoverflow.com/questions/47752324/matrix-multiplication-on-4d-numpy-arrays
# Part 1: pre-compute L operators


def spin_operators_unvectorized(nspins):
    filename = f'unvectorized{nspins}.npy'
    try:
        L = np.load(filename)
        return L
    except FileNotFoundError:
        print(f'creating unvectorized{nspins}.npy')
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
    with open(filename, 'wb') as f:
        np.save(f, L)

    return L


# Part 2: vectorized Hamiltonian using spin_operators
# First: an unvectorized Hamiltonian using spin_operators



# @timefn
def hamiltonian_unvectorized(v, J):
    nspins = len(v)
    L = spin_operators_unvectorized(nspins)
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
def spin_operators_vectorized(nspins):
    filename_Lz = f'vectorized_Lz{nspins}.npy'
    filename_Lproduct = f'vectorized_Lproduct{nspins}.npy'
    try:
        Lz = np.load(filename_Lz)
        Lproduct = np.load(filename_Lproduct)
        return Lz, Lproduct
    except FileNotFoundError:
        print(f'creating vectorized{nspins}.npy')
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
    L_T = L.transpose(1, 0, 2, 3)
    Lproduct = np.tensordot(L_T, L, axes=((1, 3), (0, 2))).swapaxes(1, 2)
    Lz = [csr_matrix(z) for z in L[2]]
    # for i in range(nspins):
    #     for j in range(nspins):
    #         Lproduct[i, j] = csr_matrix(Lproduct[i, j])
    # Lproduct_sparse = csr_matrix(Lproduct)
    with open(filename_Lz, 'wb') as f:
        np.save(f, L[2])
    with open(filename_Lproduct, 'wb') as f:
        np.save(f, Lproduct)

    return L[2], Lproduct


# @timefn
def hamiltonian_vectorized(v, J):
    nspins = len(v)
    Lz, Lproduct = spin_operators_vectorized(nspins)
    H = np.tensordot(v, Lz, axes=1)
    # L_T = L.transpose(1, 0, 2, 3)
    # Lproduct = np.tensordot(L_T, L, axes=((1, 3), (0, 2))).swapaxes(1, 2)
    scalars = 0.5 * J
    H += np.tensordot(scalars, Lproduct, axes=2)
    return H


def so_sparse(nspins):
    filename_Lz = f'sparse_Lz{nspins}.npz'
    filename_Lproduct = f'sparse_Lproduct{nspins}.npz'
    try:
        Lz = sparse.load_npz(filename_Lz)
        Lproduct = sparse.load_npz(filename_Lproduct)
        return Lz, Lproduct
    except FileNotFoundError:
        print(f'creating vectorized{nspins}.npy')
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
    L_T = L.transpose(1, 0, 2, 3)
    Lproduct = np.tensordot(L_T, L, axes=((1, 3), (0, 2))).swapaxes(1, 2)
    Lz_sparse = sparse.COO(L[2])
    # for i in range(nspins):
    #     for j in range(nspins):
    #         Lproduct[i, j] = csr_matrix(Lproduct[i, j])
    Lproduct_sparse = sparse.COO(Lproduct)
    sparse.save_npz(filename_Lz, Lz_sparse)
    sparse.save_npz(filename_Lproduct, Lproduct_sparse)

    return Lz_sparse, Lproduct_sparse


def hamiltonian_sparse(v, J):
    nspins = len(v)
    Lz, Lproduct = so_sparse(nspins)
    H = sparse.tensordot(v, Lz, axes=1)
    # L_T = L.transpose(1, 0, 2, 3)
    # Lproduct = np.tensordot(L_T, L, axes=((1, 3), (0, 2))).swapaxes(1, 2)
    scalars = 0.5 * J
    H += sparse.tensordot(scalars, Lproduct, axes=2)
    return H


if __name__ == '__main__':
    from simulation_data import spin8
    # from tests.prepare import standard_H
    v, J = spin8()
    # L = spin_operators(8)
    hamiltonians = [
                    kuprov_H(v, J),
                    kuprov_cached(v, J),
                    # hamiltonian_slow(v, J),
                    hamiltonian(v, J),
                    hamiltonian_unvectorized(v, J),
                    hamiltonian_vectorized(v, J),
                    hamiltonian_sparse(v, J)
                    ]
    for i in range(len(hamiltonians)-1):
        try:
            assert np.array_equal(hamiltonians[i], hamiltonians[i+1]), 'fail at ' + str(i)
            print('Passed: ', str(i), str(i+1))
        except AssertionError:
            print('failure at ', str(i))
            print(hamiltonians[i])
            print(hamiltonians[i + 1])
