"""A translation of Kuprov's algorithms to Python."""

import numpy as np
from scipy.sparse import csr_matrix, kron, lil_matrix

from .speedutils import timefn


@timefn
def kuprov_H_csr(v, J):

    sigma_x = csr_matrix([[0, 1 / 2], [1 / 2, 0]])
    sigma_y = csr_matrix([[0, -1j / 2], [1j / 2, 0]])
    sigma_z = csr_matrix([[1 / 2, 0], [0, -1 / 2]])
    unit = csr_matrix([[1, 0], [0, 1]])

    nspins = len(v)
    Lx = []
    Ly = []
    Lz = []

    for n in range(nspins):
        Lx_current = 1
        Ly_current = 1
        Lz_current = 1

        for k in range(nspins):
            if k == n:
                Lx_current = kron(Lx_current, sigma_x)
                Ly_current = kron(Ly_current, sigma_y)
                Lz_current = kron(Lz_current, sigma_z)
            else:
                Lx_current = kron(Lx_current, unit)
                Ly_current = kron(Ly_current, unit)
                Lz_current = kron(Lz_current, unit)

        Lx.append(Lx_current)
        Ly.append(Ly_current)
        Lz.append(Lz_current)

    H = csr_matrix((2**nspins, 2**nspins))
    for n in range(nspins):
        H += v[n] * Lz[n]

    for n in range(nspins):
        for k in range(nspins):
            if n != k:
                H += 0.5 * J[n, k] * (Lx[n]*Lx[k] + Ly[n]*Ly[k] + Lz[n]*Lz[k])

    return H.todense()


@timefn
def kuprov_H_lil(v, J):
    sigma_x = lil_matrix([[0, 1 / 2], [1 / 2, 0]])
    sigma_y = lil_matrix([[0, -1j / 2], [1j / 2, 0]])
    sigma_z = lil_matrix([[1 / 2, 0], [0, -1 / 2]])
    unit = lil_matrix([[1, 0], [0, 1]])

    nspins = len(v)
    Lx = []
    Ly = []
    Lz = []

    for n in range(nspins):
        Lx_current = 1
        Ly_current = 1
        Lz_current = 1

        for k in range(nspins):
            if k == n:
                Lx_current = kron(Lx_current, sigma_x)
                Ly_current = kron(Ly_current, sigma_y)
                Lz_current = kron(Lz_current, sigma_z)
            else:
                Lx_current = kron(Lx_current, unit)
                Ly_current = kron(Ly_current, unit)
                Lz_current = kron(Lz_current, unit)

        Lx.append(Lx_current)
        Ly.append(Ly_current)
        Lz.append(Lz_current)

    H = lil_matrix((2**nspins, 2**nspins))
    for n in range(nspins):
        H += v[n] * Lz[n]

    for n in range(nspins):
        for k in range(nspins):
            if n != k:
                H += 0.5 * J[n, k] * (Lx[n]*Lx[k] + Ly[n]*Ly[k] + Lz[n]*Lz[k])

    return H.todense()


@timefn
def kuprov_H_dense(v, J):
    sigma_x = np.matrix([[0, 1 / 2], [1 / 2, 0]])
    sigma_y = np.matrix([[0, -1j / 2], [1j / 2, 0]])
    sigma_z = np.matrix([[1 / 2, 0], [0, -1 / 2]])
    unit = np.matrix([[1, 0], [0, 1]])

    nspins = len(v)
    Lx = []
    Ly = []
    Lz = []

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

        Lx.append(Lx_current)
        Ly.append(Ly_current)
        Lz.append(Lz_current)

    H = np.zeros((2**nspins, 2**nspins), dtype=np.complex128)
    for n in range(nspins):
        H += v[n] * Lz[n]

    for n in range(nspins):
        for k in range(nspins):
            if n != k:
                H += 0.5 * J[n, k] * (Lx[n]*Lx[k] + Ly[n]*Ly[k] + Lz[n]*Lz[k])

    return H



if __name__ == '__main__':
    from simulation_data import spin8

    # v = [430, 265, 300]
    # J = csr_matrix((3, 3))
    # J[0, 1] = 7
    # J[0, 2] = 15
    # J[1, 2] = 1.5
    # J = J + J.T

    # H = kuprov_H_dense(v, J)
    # print(H)
    # H = kuprov_H_lil(*spin8())
    # print(H)
    v, J = spin8()
    for H in [kuprov_H_csr, kuprov_H_lil, kuprov_H_dense]:
        print(H(v, J))