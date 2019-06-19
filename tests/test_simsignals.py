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
import line_profiler
profile = line_profiler.LineProfiler()
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
H_RIOUX_SPARSE = hamiltonian_sparse(*rioux())


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
    # consistent with 11-spin e.g. 14.4 vs. 17.0 s, x5
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
    assert not np.allclose(V1[:, 1], V2[:, 1])
    assert np.allclose(V1_asarray[:, 1], V2[:, 1])
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


@profile
def newer_simsignals(H, nspins):
    """new_simsignals plus faster transition matrix"
    """
    m = 2 ** nspins
    E, V = np.linalg.eigh(H)
    # T = new_transition_matrix(m)
    T = cache_tm(m)
    I = np.square(V.T.dot(T.dot(V)))
    spectrum = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            if I[i, j] > 0.01:  # consider making this minimum intensity
                # cutoff a function arg, for flexibility
                v = abs(E[i] - E[j])
                spectrum.append((v, I[i, j]))
    return spectrum


# Splitting simsignals up to test vectorization of for loop
def intensity_and_energy(H, nspins):
    """Calculate intensity matrix and energies (eigenvalues) from Hamiltonian.

    Parameters
    ----------
    H (numpy.ndarray): Spin Hamiltonian
    nspins: number of spins in spin system

    Returns
    -------
    (I, E) (numpy.ndarray, numpy.ndarray) tuple of:
        I: (relative) intensity matrix
        V: 1-D array of relative energies.
    """
    m = 2 ** nspins
    E, V = np.linalg.eigh(H)
    V = V.real
    T = cache_tm(m)
    I = np.square(V.T.dot(T.dot(V)))
    return I, E


def old_compile_spectrum(I, E):
    spectrum = []
    m = I.shape[0]
    for i in range(m - 1):
        for j in range(i + 1, m):
            if I[i, j] > 0.01:  # consider making this minimum intensity
                # cutoff a function arg, for flexibility
                v = abs(E[i] - E[j])
                spectrum.append((v, I[i, j]))
    return spectrum


def new_compile_spectrum(I, E):
    I_upper = np.triu(I)
    E_matrix = np.abs(E[:, np.newaxis] - E)
    E_upper = np.triu(E_matrix)
    combo = np.stack([E_upper, I_upper])
    iv = combo.reshape(2, I.shape[0] ** 2).T

    return iv[iv[:, 1] >= 0.01]


def tupleize(array):
    newtuple = []
    for row in array:
        newtuple.append(tuple(row))
    return newtuple


def test_new_compile_spectrum():
    H = H_RIOUX
    old_spectrum = simsignals(H, 3)
    I, E = intensity_and_energy(H, 3)
    new_spec_array = new_compile_spectrum(I, E)
    new_spectrum = new_spec_array #tupleize(new_spec_array)
    print(old_spectrum[:10])
    print(new_spectrum[:10])
    assert np.allclose(old_spectrum, new_spectrum)


def vectorized_simsignals(H, nspins):
    I, E = intensity_and_energy(H, nspins)
    return new_compile_spectrum(I, E)


def test_vectorized_simsignals():
    H = H_RIOUX_SPARSE
    old_spectrum = simsignals(H, 3)
    new_spectrum = vectorized_simsignals(H, 3)
    assert np.allclose(old_spectrum, new_spectrum)


def difference_matrix(array):
    """From a 1D array, compute a 2D array of differences between all
    elements.
    Ref: https://stackoverflow.com/questions/9704565/populate-numpy-matrix-from-the-difference-of-two-vectors
    """
    return np.abs(array[:, np.newaxis] - array)

def test_difference_matrix():
    array = np.array([1, 5])
    assert np.array_equal(
        np.array([[0, 4], [4, 0]]),
        difference_matrix(array)
    )

# def stack_matrix(m1, m2):
#     return np.dstack((m1, m2))
#
#
# def test_stack_matrix():
#     m1 = [[100, 200], [300, 400]]
#     m2 = [[1, 2], [3, 4]]
#     expected = np.array([
#         [[100, 200], [300, 400]],
#         [[1, 2], [3, 4]]
#     ])
#     obtained = stack_matrix(m1, m2)
#     print(obtained)
#     assert np.array_equal(expected, obtained)


def test_split_simsignals():
    H = H_RIOUX
    refspec = newer_simsignals(H, 3)
    I, E = intensity_and_energy(H, 3)
    print(I)
    testspec = old_compile_spectrum(I, E)
    assert np.array_equal(refspec, testspec)


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


@timefn
def loop_newer_simsignals(H, nspins, n):
    for i in range(n):
        _ = newer_simsignals(H, nspins)
    return _


@timefn
def new_transition_matrix(n):
    """Dense/optimized(?) version of transition matrix."""
    """
    Creates a matrix of allowed transitions.

    The integers 0-`n`, in their binary form, code for a spin state
    (alpha/beta). The (i,j) cells in the matrix indicate whether a transition
    from spin state i to spin state j is allowed or forbidden.
    See the ``is_allowed`` function for more information.

    Parameters
    ---------
    n : dimension of the n,n matrix (i.e. number of possible spin states).

    Returns
    -------
    csr_matrix
        a transition matrix that can be used to compute the intensity of
    allowed transitions.

    Note: lil_matrix becomes csr_matrix after adding transpose.
    """
    # Testing with spin-11 and the new simsignals: new_transition matrix is
    # faster than transition_matrix (0.8 vs 1.1 s), but simsignals with
    # matrix is only slightly faster, and simsignals with ndarray (the
    # future) is slower. Can try compromising by keeping lil_matrix but
    # condensing popcount/is_allowed, but best may be just to cache these.
    T = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            if bin(i ^ j).count('1') == 1:
                T[i, j] = 1
    # T = T + T.T
    T += T.T
    return T


# @timefn
def cache_tm(n):
    """spin11 test indicates this leads to faster overall simsignals().

    11 spin x 6: 29.6 vs. 35.1 s
    8 spin x 60: 2.2 vs 3.0 s"""
    filename = f'transitions{n}.npz'
    try:
        T = sparse.load_npz(filename)
        return T
    except FileNotFoundError:
        print(f'creating {filename}')
        T = np.zeros((n, n))
        for i in range(n - 1):
            for j in range(i + 1, n):
                if bin(i ^ j).count('1') == 1:
                    T[i, j] = 1
        # T = T + T.T
        T += T.T
        T_sparse = sparse.COO(T)
        sparse.save_npz(filename, T_sparse)
        return T_sparse


def test_cache_tm():
    T1 = transition_matrix(2**3)
    T2 = cache_tm(2**3)
    assert np.array_equal(T1.todense(), T2.todense())


def test_tm():
    t_old = transition_matrix(2**11)
    t_new = cache_tm(2**11)
    assert np.array_equal(t_old.todense(), t_new.todense())


def test_simsignals():
    # Tests indicate new_simsignals is faster, with the difference increasing
    # at higher spin numbers.
    # matrix, spin 11, x3: old 64.5 s, new 12.1 s
    # array, spin11, x3: old 62.3 s, new 16.7 s
    n = 60
    H = H8_NDARRAY
    nspins = 8
    s1 = loop_new_simsignals(H, nspins, n)
    s2 = loop_newer_simsignals(H, nspins, n)
    assert np.allclose(s1, s2)


def test_loop_newer_simsignals():
    n = 1
    _ = loop_newer_simsignals(H11_NDARRAY, 11, n)
    assert _ is not None


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
    intensity_matrices = [f(Vrow, T, Vcol, 1).todense()
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


if __name__ == '__main__':
    newer_simsignals(H11_NDARRAY, 11)
