"""Testing the performance for the entire spectrum calculation."""
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
import sparse

from nmrtools.nmrmath import is_allowed, normalize_spectrum, transition_matrix
from nmrtools.nmrplot import nmrplot
from speedtest.compare_hamiltonians import hamiltonian, hamiltonian_sparse
from speedtest.speedutils import timefn

from simulation_data import spin8, spin3, spin11

H8 = hamiltonian(*spin8())
H8_SPARSE = hamiltonian_sparse(*spin8())
print('Import type: ', type(H8_SPARSE))


# Below are older functions that were written before completion of
# test_simsignals.py

def nspinspec(h_func, freqs, couplings, normalize=True):
    """A version of nmrtools.nmrmath.nspinspec that accepts different H functions."""
    """
    Calculates second-order spectral data (freqency and intensity of signals)
    for *n* spin-half nuclei.

    Parameters
    ---------
    freqs : [float...]
        a list of *n* nuclei frequencies in Hz
    couplings : array-like
        an *n, n* array of couplings in Hz. The order
        of nuclei in the list corresponds to the column and row order in the
        matrix, e.g. couplings[0][1] and [1]0] are the J coupling between
        the nuclei of freqs[0] and freqs[1].
    normalize: bool
        True if the intensities should be normalized so that total intensity
        equals the total number of nuclei.

    Returns
    -------
    spectrum : [(float, float)...]
        a list of (frequency, intensity) tuples.
    """
    nspins = len(freqs)
    H = h_func(freqs, couplings)
    spectrum = simsignals(H, nspins)
    if normalize:
        spectrum = normalize_spectrum(spectrum, nspins)
    return spectrum


# @timefn
def simsignals(H, nspins):
    """
    Calculates the eigensolution of the spin Hamiltonian H and, using it,
    returns the allowed transitions as list of (frequency, intensity) tuples.

    Parameters
    ---------

    H : ndarray
        the spin Hamiltonian.
    nspins : int
        the number of nuclei in the spin system.

    Returns
    -------
    spectrum : [(float, float)...]
        a list of (frequency, intensity) tuples.
    """
    """The original simsignals."""
    # This routine was optimized for speed by vectorizing the intensity
    # calculations, replacing a nested-for signal-by-signal calculation.
    # Considering that hamiltonian was dramatically faster when refactored to
    # use arrays instead of sparse matrices, consider an array refactor to this
    # function as well.

    # The eigensolution calculation apparently must be done on a dense matrix,
    # because eig functions on sparse matrices can't return all answers?!
    # Using eigh so that answers have only real components and no residual small
    # unreal components b/c of rounding errors
    E, V = np.linalg.eigh(H)    # V will be eigenvectors, v will be frequencies

    # 2019-04-27: the statement below may be wrong. May be entirely real already
    # Eigh still leaves residual 0j terms, so:
    V = np.asmatrix(V.real)

    # Calculate signal intensities
    Vcol = csc_matrix(V)
    Vrow = csr_matrix(Vcol.T)
    m = 2 ** nspins
    T = transition_matrix(m)
    I = Vrow * T * Vcol
    I = np.square(I.todense())

    spectrum = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            if I[i, j] > 0.01:  # consider making this minimum intensity
                                # cutoff a function arg, for flexibility
                v = abs(E[i] - E[j])
                spectrum.append((v, I[i, j]))

    return spectrum


@timefn
def simsignals2(H, nspins):
    """
    Calculates the eigensolution of the spin Hamiltonian H and, using it,
    returns the allowed transitions as list of (frequency, intensity) tuples.

    Parameters
    ---------

    H : ndarray
        the spin Hamiltonian.
    nspins : int
        the number of nuclei in the spin system.

    Returns
    -------
    spectrum : [(float, float)...]
        a list of (frequency, intensity) tuples.
    """
    """Refactoring simsignals to not do dense<->sparse interconversions to see if 
    it will run faster."""
    # This routine was optimized for speed by vectorizing the intensity
    # calculations, replacing a nested-for signal-by-signal calculation.
    # Considering that hamiltonian was dramatically faster when refactored to
    # use arrays instead of sparse matrices, consider an array refactor to this
    # function as well.

    # The eigensolution calculation apparently must be done on a dense matrix,
    # because eig functions on sparse matrices can't return all answers?!
    # Using eigh so that answers have only real components and no residual small
    # unreal components b/c of rounding errors

    E, V = np.linalg.eigh(H)    # V will be eigenvectors, v will be frequencies

    # Eigh still leaves residual 0j terms, so:
    # V = np.asmatrix(V.real)
    Vcol = V.real
    # Calculate signal intensities
    # Vcol = csc_matrix(V)
    # Vrow = csr_matrix(Vcol.T)
    Vrow = Vcol.T
    m = 2 ** nspins
    T = transition_matrix2(m)
    I = Vrow @ T @ Vcol
    # I = np.square(I.todense())
    I = np.square(I)
    spectrum = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            if I[i, j] > 0.01:  # consider making this minimum intensity
                                # cutoff a function arg, for flexibility
                v = abs(E[i] - E[j])
                spectrum.append((v, I[i, j]))

    return spectrum

@timefn
def transition_matrix2(n):
    """dense version"""
    T = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            if is_allowed(i, j):
                T[i, j] = 1
    T = T + T.T
    return T


@timefn
def transition_matrix_sparse(n):
    """uses sparse package"""
    T = lil_matrix((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            if is_allowed(i, j):
                T[i, j] = 1
    T = T + T.T
    return sparse.COO(T)


def test_transition_matrices():
    nspins = 11
    n = 2 ** nspins
    T_old = transition_matrix(n)
    T_dense = transition_matrix2(n)
    T_sparse = transition_matrix_sparse(n)
    assert np.array_equal(T_old.todense(), T_dense)
    assert np.array_equal(T_old.todense(), T_sparse.todense())


@timefn
def test_simsignals():
    n = 5
    for i in range(n):
        spec1 = simsignals(H8, 8)
        spec2 = simsignals(H8_SPARSE, 8)
    assert np.allclose(spec1, spec2)


@timefn
def test_simsignals2():
    n = 5
    for i in range(n):
        spec1 = simsignals2(H8, 8)
        spec2 = simsignals2(H8_SPARSE, 8)
    assert np.allclose(spec1, spec2)
    # print(type(H8), type(H8_SPARSE))


def test_oldH_profile():
    n = 10
    H = hamiltonian(*spin11())
    for i in range(n):
        _ = simsignals2(H, 11)
    assert 1 == 1


def test_newH_profile():
    n = 10
    H = hamiltonian_sparse(*spin11())
    for i in range(n):
        _ = simsignals2(H, 11)
    assert 1 == 1


@timefn
def test_nspinspec():
    v, J = spin8()
    # # print(v)
    # # print(J)
    h = hamiltonian_sparse(v, J)
    # # print(h)
    nspins = len(v)
    return simsignals(h, nspins)
    # spectrum = normalize_spectrum(spectrum, nspins)
    # return hamiltonian_sparse(v, J)
    assert 1 == 1


def test_smoke():
    nmrplot(nspinspec(hamiltonian, *spin8()))
    nmrplot(nspinspec(hamiltonian_sparse, *spin8()))


