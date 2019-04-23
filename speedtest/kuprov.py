"""A translation of Kuprov's algorithms to Python."""

import numpy as np
from scipy.sparse import csr_matrix, kron, lil_matrix
from speedtest.speedutils import timefn

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


# @timefn
def kuprov_H_dense(v, J):
    # 2019-04-21: removed deprecated np.matrix; replaced with np.array
    sigma_x = np.array([[0, 1 / 2], [1 / 2, 0]])
    sigma_y = np.array([[0, -1j / 2], [1j / 2, 0]])
    sigma_z = np.array([[1 / 2, 0], [0, -1 / 2]])
    unit = np.array([[1, 0], [0, 1]])

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
                H += 0.5 * J[n, k] * (Lx[n]@Lx[k] + Ly[n]@Ly[k] + Lz[n]@Lz[k])

    return H


def kuprov_so(nspins):
    filename = f'kuprov_so{nspins}.npy'
    try:
        spin_operators = np.load(filename)
        return spin_operators
    except FileNotFoundError:
        print(f'creating kuprov_so{nspins}.npy')

    sigma_x = np.array([[0, 1 / 2], [1 / 2, 0]])
    sigma_y = np.array([[0, -1j / 2], [1j / 2, 0]])
    sigma_z = np.array([[1 / 2, 0], [0, -1 / 2]])
    unit = np.array([[1, 0], [0, 1]])

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
    with open(filename, 'wb') as f:
        np.save(f, (Lx, Ly, Lz))
    return (Lx, Ly, Lz)


# @timefn
def kuprov_cached(v, J):
    nspins = len(v)
    Lx, Ly, Lz = kuprov_so(nspins)
    H = np.zeros((2 ** nspins, 2 ** nspins), dtype=np.complex128)
    for n in range(nspins):
        H += v[n] * Lz[n]

    for n in range(nspins):
        for k in range(nspins):
            if n != k:
                H += 0.5 * J[n, k] * (Lx[n] @ Lx[k] + Ly[n] @ Ly[k] + Lz[n] @ Lz[k])
    return H


if __name__ == '__main__':
    from simulation_data import spin8
    from tests.prepare import standard_8
    v, J = spin8()
    hamiltonians = [f(v, J) for f in [kuprov_H_csr, kuprov_H_lil, kuprov_H_dense, kuprov_cached]]
    for i, h in enumerate(hamiltonians):
        try:
            assert np.array_equal(h, standard_8), 'fail at ' + str(i)
            print('Passed: ', str(i), str(i+1))
        except AssertionError:
            print('failure at ', str(i))
            print(hamiltonians[i])
    print(standard_8)
    print(hamiltonians[-1])
