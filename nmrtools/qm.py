"""qm contains functions for the quantum-mechanical (second-order)
calculation of NMR spectra.

Because numpy.matrix is marked as deprecated, in Winter/Spring 2019 the qm
code was refactored to a) accommodate this deprecation and b) speed up the
calculations. The fastest calculations rely on:

1. the pydata/sparse library. SciPy's sparse depends on numpy.matrix,
and they currently recommend that pydata/sparse be used for now.

2. Caching partial solutions for spin operators and transition matrices as
.npz files.

If the pydata/sparse package is no longer available, and/or if distributing
the library with .npz files via PyPI is problematic, then a backup is
required. The qm module for now provides two sets of functions for
calculating second-order spectra: one using pydata/sparse and caching,
and the other using neither.
"""
import os

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import sparse
from .nmrmath import normalize_spectrum, transition_matrix

SO_DIR = os.path.join(os.path.abspath('..'), 'nmrtools', 'bin')


def so_dense(nspins):
    """
    Calculate spin operators required for constructing the spin hamiltonian.

    Parameters
    ----------
    nspins: int
        the number of spins in the spin system

    Returns
    -------
    (Lz, Lproduct): a tuple of:
        Lz: 3d array of shape (n, 2^n, 2^n) representing [Lz1, Lz2, ...Lzn]
        Lproduct: 4d array of shape (n, n, 2^n, 2^n), representing an n x n
        array (cartesian product) for all combinations of
        Lxa*Lxb + Lya*Lyb + Lza*Lzb, where 1 <= a, b <= n.
    """
    sigma_x = np.array([[0, 1 / 2], [1 / 2, 0]])
    sigma_y = np.array([[0, -1j / 2], [1j / 2, 0]])
    sigma_z = np.array([[1 / 2, 0], [0, -1 / 2]])
    unit = np.array([[1, 0], [0, 1]])

    L = np.empty((3, nspins, 2 ** nspins, 2 ** nspins),
                 dtype=np.complex128)  # consider other dtype?
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

    # ref: https://stackoverflow.com/questions/47752324/matrix-multiplication-on-4d-numpy-arrays
    L_T = L.transpose(1, 0, 2, 3)
    Lproduct = np.tensordot(L_T, L, axes=((1, 3), (0, 2))).swapaxes(1, 2)

    return L[2], Lproduct


def so_sparse(nspins):
    """Either load a presaved set of spin operators as numpy arrays, or
    calculate them and save them if a presaved set wasn't found.

    Parameters
    ----------
    nspins: int
        the number of spins in the spin system

    Returns
    -------
    (Lz, Lproduct): a tuple of:
        Lz: 3d sparse.COO array of shape (n, 2^n, 2^n) representing
        [Lz1, Lz2, ...Lzn]
        Lproduct: 4d sparse.COO array of shape (n, n, 2^n, 2^n), representing
        an n x n array (cartesian product) for all combinations of
        Lxa*Lxb + Lya*Lyb + Lza*Lzb, where 1 <= a, b <= n.

    Side Effect
    -----------
    Saves the results as .npz files to the bin directory if they were not
    found there.
    """
    filename_Lz = f'Lz{nspins}.npz'
    filename_Lproduct = f'Lproduct{nspins}.npz'
    bin_dir = os.path.join(os.path.dirname(__file__), 'bin')
    path_Lz = os.path.join(bin_dir, filename_Lz)
    path_Lproduct = os.path.join(bin_dir, filename_Lproduct)

    try:
        Lz = sparse.load_npz(path_Lz)
        Lproduct = sparse.load_npz(path_Lproduct)
        return Lz, Lproduct
    except FileNotFoundError:
        print('no SO file ', filename_Lz, ' found in: ', bin_dir)
        print(f'creating {filename_Lz} and {filename_Lproduct}')
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
    sparse.save_npz(path_Lz, Lz_sparse)
    sparse.save_npz(path_Lproduct, Lproduct_sparse)

    return Lz_sparse, Lproduct_sparse


def hamiltonian_dense(v, J):
    nspins = len(v)
    Lz, Lproduct = so_dense(nspins)
    H = np.tensordot(v, Lz, axes=1)
    scalars = 0.5 * J
    H += np.tensordot(scalars, Lproduct, axes=2)
    return H


def hamiltonian_sparse(v, J):
    """

        Parameters
        ----------
        v
        J

        Returns
        -------
        H: a numpy.ndarray, and NOT a sparse.COO?!?!?!
        """
    nspins = len(v)
    Lz, Lproduct = so_sparse(nspins)
    H = sparse.tensordot(v, Lz, axes=1)
    scalars = 0.5 * J
    H += sparse.tensordot(scalars, Lproduct, axes=2)
    return H


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


def nspinspec_dense(freqs, couplings, normalize=True):
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
    H = hamiltonian_dense(freqs, couplings)
    spectrum = simsignals(H, nspins)
    if normalize:
        spectrum = normalize_spectrum(spectrum, nspins)
    return spectrum


def cache_tm(nspins):
    """

    Parameters
    ----------
    nspins

    Returns
    -------

    """
    """spin11 test indicates this leads to faster overall simsignals().

    11 spin x 6: 29.6 vs. 35.1 s
    8 spin x 60: 2.2 vs 3.0 s"""
    filename = f'T{nspins}.npz'
    path = os.path.join(SO_DIR, filename)
    try:
        T = sparse.load_npz(path)
        return T
    except FileNotFoundError:
        print(f'creating {filename}')
        n = 2 ** nspins
        T = np.zeros((n, n))
        for i in range(n - 1):
            for j in range(i + 1, n):
                if bin(i ^ j).count('1') == 1:
                    T[i, j] = 1
        # T = T + T.T
        T += T.T
        T_sparse = sparse.COO(T)
        sparse.save_npz(path, T_sparse)
        return T_sparse


def intensity_and_energy(H, nspins):
    """Calculate intensity matrix and energies (eigenvalues) from Hamiltonian.

    Parameters
    ----------
    H:  numpy.ndarray
        Spin Hamiltonian
    nspins: int
        number of spins in spin system

    Returns
    -------
    (I, E): (numpy.ndarray, numpy.ndarray) tuple of:
        I: (relative) intensity matrix
        V: 1-D array of relative energies.
    """
    E, V = np.linalg.eigh(H)
    V = V.real
    T = cache_tm(nspins)
    I = np.square(V.T.dot(T.dot(V)))
    return I, E


def new_compile_spectrum(I, E):
    I_upper = np.triu(I)
    E_matrix = np.abs(E[:, np.newaxis] - E)
    E_upper = np.triu(E_matrix)
    combo = np.stack([E_upper, I_upper])
    iv = combo.reshape(2, I.shape[0] ** 2).T

    return iv[iv[:, 1] >= 0.01]


def vectorized_simsignals(H, nspins):
    I, E = intensity_and_energy(H, nspins)
    return new_compile_spectrum(I, E)


def nspinspec_sparse(freqs, couplings, normalize=True):
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
    H = hamiltonian_sparse(freqs, couplings)
    spectrum = vectorized_simsignals(H, nspins)
    if normalize:
        spectrum = normalize_spectrum(spectrum, nspins)
    return spectrum
