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


def test_matrix():
    # conclusion: matrix faster with np eigh (ca 5-6 s vs. 8-9 s) 8-spin
    # consistent with 11-spin e.g. 6.4 vs. 8.5, x5
    multitest(np.linalg.eigh, H11_MATRIX, 5)
    multitest(scipy.linalg.eigh, H11_MATRIX, 5)
    assert 1 == 1


def test_array():
    # conclusion: ndarray faster in np eigh than scipy eigh
    # (9.8 vs 13.3 s, 10.3 vs 13.5, 10.6 vs. 13.9) 8-spin
    # consistemt with 11-spin e.g. 14.4 vs. 17.0 s, x5
    # conclusion: matrix faster than ndarray, and np faster than scipy for eigh
    multitest(np.linalg.eigh, H11_NDARRAY, 5)
    multitest(scipy.linalg.eigh, H11_NDARRAY, 5)
    assert 1 == 1

def matrix_eigh(h, n):
    for i in range(n):
        E1, V1 = numpy_eigh(h)
        E2, V2 = scipy_eigh(h)
    return (E1, V1), (E2, V2)


def test_ev_matching():
    numpy_result, scipy_result = matrix_eigh(H_RIOUX, 1)
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


def test_matrix_eigh():
    matrix_eigh(H11_MATRIX, 1)
    matrix_eigh(H11_NDARRAY, 1)
    assert 1 == 1





