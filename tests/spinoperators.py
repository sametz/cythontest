"""Looked at making spin operator calculation faster, or allowing it to make
solutions for 12+ spins, by using the sparse package and/or numba. Lack of
some numpy functionality in sparse (e.g. assign in place; swapaxes) and numba
limitations for types makes this look like a non-productive place to seek
speed gains, esp. if the spin operators will indeed be distributed as
binaries along with the nmrtools package.
"""
from numba import njit, jit
import numpy as np
import sparse

from speedtest.speedutils import timefn


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
    L_T = L.transpose(1, 0, 2, 3)
    Lproduct = np.tensordot(L_T, L, axes=((1, 3), (0, 2))).swapaxes(1, 2)
    Lz_sparse = sparse.COO(L[2])
    Lproduct_sparse = sparse.COO(Lproduct)

    return Lz_sparse, Lproduct_sparse


def so_sparse(nspins):
    """Attempted to sparsify the spin operator generation, but there are hurdles. In-place assignments can't be done on
    sparse matrices. Apparently no .swapaxes method for COO matrix either.

    """
    sigma_x = sparse.COO(np.array([[0, 1 / 2], [1 / 2, 0]]))
    sigma_y = sparse.COO(np.array([[0, -1j / 2], [1j / 2, 0]]))
    sigma_z = sparse.COO(np.array([[1 / 2, 0], [0, -1 / 2]]))
    unit = sparse.COO(np.array([[1, 0], [0, 1]]))

    L = np.empty((3, nspins, 2 ** nspins, 2 ** nspins), dtype=np.complex128)  # consider other dtype?
    # Lxs = []
    # Lys = []
    # Lzs = []

    for n in range(nspins):
        Lx_current = 1
        Ly_current = 1
        Lz_current = 1

        for k in range(nspins):
            if k == n:
                Lx_current = sparse.kron(Lx_current, sigma_x)
                Ly_current = sparse.kron(Ly_current, sigma_y)
                Lz_current = sparse.kron(Lz_current, sigma_z)
            else:
                Lx_current = sparse.kron(Lx_current, unit)
                Ly_current = sparse.kron(Ly_current, unit)
                Lz_current = sparse.kron(Lz_current, unit)

        # Lxs[n] = Lx_current
        # Lys[n] = Ly_current
        # Lzs[n] = Lz_current
        # print(Lx_current.todense())
        L[0][n] = Lx_current.todense()
        L[1][n] = Ly_current.todense()
        L[2][n] = Lz_current.todense()
    Lz_sparse = sparse.COO(L[2])
    L_T = L.transpose(1, 0, 2, 3)
    L_sparse = sparse.COO(L)
    L_T_sparse = sparse.COO(L_T)
    Lproduct = sparse.tensordot(L_T_sparse, L_sparse, axes=((1, 3), (0, 2))).swapaxes(1, 2)
    # Lz_sparse = sparse.COO(L[2])
    Lproduct_sparse = sparse.COO(Lproduct)

    return Lz_sparse, Lproduct_sparse


@jit
def so_numba(nspins):
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
    Lproduct_sparse = sparse.COO(Lproduct)

    return Lz_sparse, Lproduct_sparse


@njit
def so_numba_old(nspins):
    sigma_x = np.array([[0, 1 / 2], [1 / 2, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j / 2], [1j / 2, 0]], dtype=np.complex128)
    sigma_z = np.array([[1 / 2, 0], [0, -1 / 2]], dtype=np.complex128)
    unit = np.array([[1, 0], [0, 1]], dtype=np.complex128)

    L = np.empty((3, nspins, 2 ** nspins, 2 ** nspins), dtype=np.complex128)  # consider other dtype?
    for n in range(nspins):
        Lx_current = sigma_x if n == 0 else unit
        Ly_current = sigma_y if n == 0 else unit
        Lz_current = sigma_z if n == 0 else unit

        for k in range(1, nspins):
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
    # Lz = [csr_matrix(z) for z in L[2]]
    # for i in range(nspins):
    #     for j in range(nspins):
    #         Lproduct[i, j] = csr_matrix(Lproduct[i, j])
    # Lproduct_sparse = csr_matrix(Lproduct)
    # with open(filename_Lz, 'wb') as f:
    #     np.save(f, L[2])
    # with open(filename_Lproduct, 'wb') as f:
    #     np.save(f, Lproduct)

    return L[2], Lproduct


@jit
def so_classic(nspins):
    sigma_x = np.array([[0, 1 / 2], [1 / 2, 0]])
    sigma_y = np.array([[0, -1j / 2], [1j / 2, 0]])
    sigma_z = np.array([[1 / 2, 0], [0, -1 / 2]])
    unit = np.array([[1, 0], [0, 1]])

    # The following empty arrays will be used to store the
    # Cartesian spin operators.
    Lx = np.empty((nspins), dtype='object')
    Ly = np.empty((nspins), dtype='object')
    Lz = np.empty((nspins), dtype='object')

    for n in range(nspins):
        Lx[n] = 1
        Ly[n] = 1
        Lz[n] = 1
        for k in range(nspins):
            if k == n:  # Diagonal element
                Lx[n] = np.kron(Lx[n], sigma_x)
                Ly[n] = np.kron(Ly[n], sigma_y)
                Lz[n] = np.kron(Lz[n], sigma_z)
            else:  # Off-diagonal element
                Lx[n] = np.kron(Lx[n], unit)
                Ly[n] = np.kron(Ly[n], unit)
                Lz[n] = np.kron(Lz[n], unit)

    #     print('Lx: ', Lx)
    #     print('Ly: ', Ly)
    #     print('Lz: ', Lz)
    Lcol = np.vstack((Lx, Ly, Lz)).real
    Lrow = Lcol.T  # As opposed to sparse version of code, this works!
    #     print('Lcol: ', Lcol.shape)
    #     print(Lcol)
    #     print('Lrow: ', Lrow.shape)
    #     print(Lrow)
    Lproduct = np.dot(Lrow, Lcol)

    # Lz_sparse = [csr_matrix(z) for z in Lz]
    #
    # for i in range(nspins):
    #     for j in range(nspins):
    #         Lproduct[i, j] = csr_matrix(Lproduct[i, j])

    #     Lproduct_sparse = csr_matrix(Lproduct)
    #     print(Lz_sparse)
    # print(Lproduct)
    # so_save(nspins, Lz_sparse, Lproduct)

    return Lz, Lproduct


@timefn
def so_loop(nspins, n):
    for i in range(n):
        _ = spin_operators(nspins)
    return _


@timefn
def numba_loop(nspins, n):
    for i in range(n):
        _ = so_classic(nspins)
    return _


if __name__ == '__main__':
    Lz1, Lprod1 = spin_operators(3)
    Lz2, Lprod2 = so_numba(3)
    # print(Lz1.todense())
    # print(Lz2.todense())
    assert np.array_equal(Lz1.todense(), Lz2.todense()), 'Z mismatch'
    assert np.array_equal(Lprod1.todense(), Lprod2.todense()), 'P mismatch'
    so_loop(3, 50)
    numba_loop(3, 50)

