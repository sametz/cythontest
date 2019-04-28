"""Currently, simsignals is a bottleneck. This code is for optimizing its
performance.

Hypotheses:
    - the Cartesian product of Lrow @ Lcol is the bottleneck
    - deprecated numpy.matrix will work better with scipy.sparse;
    numpy.ndarray will work better with sparse.COO
    -minimizing interconversions between formats is desirable
    - sparse.COO may not have an advantage over scipy.sparse for 1- and 2-D
    matrices.

Notes:
    np.linalg.eigh returns v as ndarray if input is ndarray, matrix if matrix;
    np.scipy.eigh returns v as ndarray
"""
import numpy as np
import scipy
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
import sparse

from nmrtools.nmrmath import is_allowed, normalize_spectrum, transition_matrix
from nmrtools.nmrplot import nmrplot
from speedtest.compare_hamiltonians import hamiltonian, hamiltonian_sparse
from speedtest.speedutils import timefn
from tests.test_spectraspeed import simsignals

from simulation_data import spin8, spin3, spin11, rioux

H3_MATRIX = hamiltonian(*spin3())
H3_NDARRAY = hamiltonian_sparse(*spin3())
H8_MATRIX = hamiltonian(*spin8())
H8_NDARRAY = hamiltonian_sparse(*spin8())
H11_MATRIX = hamiltonian(*spin11())
H11_NDARRAY = hamiltonian_sparse(*spin11())
H_RIOUX = hamiltonian(*rioux())


@timefn
def numpy_eigh(h):
    return np.linalg.eigh(h)


@timefn
def scipy_eigh(h):
    return scipy.linalg.eigh(h)


@timefn
def multitest(test, h, n):
    for i in range(n):
        test(h)


def test_matrix_eigh_time():
    # conclusion: matrix faster with np eigh (ca 5-6 s vs. 8-9 s) 8-spin
    # consistent with 11-spin e.g. 6.4 vs. 8.5, x5
    multitest(np.linalg.eigh, H11_MATRIX, 5)
    multitest(scipy.linalg.eigh, H11_MATRIX, 5)
    assert 1 == 1


def test_array_eigh_time():
    # conclusion: ndarray faster in np eigh than scipy eigh
    # (9.8 vs 13.3 s, 10.3 vs 13.5, 10.6 vs. 13.9) 8-spin
    # consistemt with 11-spin e.g. 14.4 vs. 17.0 s, x5
    # conclusion: matrix faster than ndarray, and np faster than scipy for eigh
    multitest(np.linalg.eigh, H11_NDARRAY, 5)
    multitest(scipy.linalg.eigh, H11_NDARRAY, 5)
    assert 1 == 1

def loop_numpy_scipy_eigensolutions(h, n):
    for i in range(n):
        E1, V1 = numpy_eigh(h)
        E2, V2 = scipy_eigh(h)
    return (E1, V1), (E2, V2)


def test_eigenvectors_real():
    # GIVEN sets of eigenvectors for candidate hamiltonians and eigh functions
    matrix_numpy, matrix_scipy = loop_numpy_scipy_eigensolutions(H8_MATRIX, 1)
    array_numpy, array_scipy = loop_numpy_scipy_eigensolutions(H8_NDARRAY, 1)
    _, V_matrix_numpy = matrix_numpy
    _, V_matrix_scipy = matrix_scipy
    _, V_array_numpy = array_numpy
    _, V_array_scipy = array_scipy
    # WHEN they are compared to their real components
    V = [V_matrix_numpy, V_matrix_scipy, V_array_numpy, V_array_scipy]
    V_real = [i.real for i in V]
    for i, j in zip(V, V_real):
        # THEN they are identical (i.e. there were no imaginary parts to
        # discard)
        assert np.array_equal(i, j)


def test_ev_matching():
    numpy_result, scipy_result = loop_numpy_scipy_eigensolutions(H_RIOUX, 1)
    E1, V1 = numpy_result
    V1_asarray = np.array(V1)
    print(type(V1))
    print(type(V1_asarray))
    E2, V2 = scipy_result
    print(type(V2))
    assert np.allclose(E1, E2)
    # accessing eigenvectors is different depending on whether V is a matrix
    # or an ndarray
    assert not np.allclose(V1[:,1], V2[:, 1])
    assert np.allclose(V1_asarray[:,1], V2[:,1])
    # The different eigenvector solutions have terms differing in sign, which
    # apparently is OK. So, testing functional equality by using squares::
    V1_ = np.square(V1)
    V2_ = np.square(V2)
    # print(V1_)
    # print(V2_)
    assert np.allclose(V1_, V2_)


@timefn
def original_intensity_matrix(V, T, n):
    for i in range(n):
        Vcol = csc_matrix(V)
        Vrow = csr_matrix(Vcol.T)
        I =  Vrow.dot(T.dot(Vcol))
        I = I.power(2)
    print(type(I))
    return I


@timefn
def norow_intensity_matrix(V, T, n):
    for i in range(n):
        Vcol = csc_matrix(V)
        I = Vcol.T.dot(T.dot(Vcol))
        I = I.power(2)
    print(type(I))
    return I


@timefn
def dense_intensity_matrix(V, T, n):
    for i in range(n):
        I = V.T.dot(T.dot(V))
        I = np.square(I)
    print(type(I))
    return I


def test_new_matrix_simsignals():
    # Testing with H11_MATRIX: dense is MUCH faster!
    # e.g. 13.8 vs. 13.6 vs. 0.28 s original/norow/dense
    # H11_NDARRAY 22.9 / 23.2 / 0.57 s
    n = 1
    nspins = 11
    E, V = np.linalg.eigh(H11_NDARRAY)
    T = transition_matrix(2 ** nspins)
    test_functions = [original_intensity_matrix,
                      norow_intensity_matrix,
                      dense_intensity_matrix]
    intensity_matrices = []
    intensity_matrices.append((original_intensity_matrix(V, T, n).todense()))
    intensity_matrices.append((norow_intensity_matrix(V, T, n).todense()))
    intensity_matrices.append(dense_intensity_matrix(V, T, n))
    assert np.allclose(intensity_matrices[0], intensity_matrices[1])
    assert np.allclose(intensity_matrices[1], intensity_matrices[2])


# @timefn
def new_simsignals(H, nspins):
    """Taking lessons from test results to create a faster simsignals
    function.
    """
    m = 2 ** nspins
    E, V = np.linalg.eigh(H)
    T = transition_matrix(m)
    I = np.square(V.T.dot(T.dot(V)))
    spectrum = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            if I[i, j] > 0.01:  # consider making this minimum intensity
                # cutoff a function arg, for flexibility
                v = abs(E[i] - E[j])
                spectrum.append((v, I[i, j]))
    return spectrum


@timefn
def loop_simsignals(H, nspins, n):
    for i in range(n):
        _ = simsignals(H, nspins)
    return _


@timefn
def loop_new_simsignals(H, nspins, n):
    for i in range(n):
        _ = new_simsignals(H, nspins)
    return _


def test_simsignals():
    # Tests indicate new_simsignals is faster, with the difference increasing
    # at higher spin numbers.
    # matrix, spin 11, x3: old 64.5 s, new 12.1 s
    # array, spin11, x3: old 62.3 s, new 16.7 s
    n = 3
    H = H11_NDARRAY
    nspins = 11
    s1 = loop_simsignals(H, nspins, n)
    s2 = loop_new_simsignals(H, nspins, n)
    assert np.allclose(s1, s2)


@timefn
def istar(a, b, c, n):
    for i in range(n):
        _ = a * b * c
    return _


@timefn
def iat(a, b, c, n):
    for i in range(n):
        _ = a @ b @ c
    return _


@timefn
def idot(a, b, c, n):
    for i in range(n):
        _ = a.dot(b.dot(c))
    return _


def test_dot_speed():
    # Test suggests .dot has a modest speed advantage.
    # 10000 runs: 276s/298s/252s spin-8 star/at/dot
    E, V = np.linalg.eigh(H8_MATRIX)
    Vcol = csc_matrix(V)
    Vrow = csr_matrix(Vcol.T)
    T = transition_matrix(2**8)
    intensity_matrices = [f(Vrow, T, Vcol, 10000).todense()
                          for f in [istar, iat, idot]]
    for i in range(2):
        assert np.allclose(intensity_matrices[i], intensity_matrices[i+1])


def test_matrix_multiplication():
    """A sanity test that all three versions of matrix multiplication are
    identical.
    """
    E, V = np.linalg.eigh(H_RIOUX)
    Vcol = csc_matrix(V)
    Vrow = csr_matrix(Vcol.T)
    T = transition_matrix(8)
    Istar = Vrow * T * Vcol
    Iat = Vrow @ T @ Vcol
    Idot = Vrow.dot(T.dot(Vcol))
    assert np.allclose(Istar.todense(), Iat.todense())
    assert np.allclose(Istar.todense(), Idot.todense())



