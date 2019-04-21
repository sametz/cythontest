"""Hacking in references for testing"""
import numpy as np

from simulation_data import spin8, spin3


def hamiltonian_original(freqlist, couplings):
    """
    Computes the spin Hamiltonian for `n` spin-1/2 nuclei.

    Parameters
    ---------
    freqlist : array-like
        a list of frequencies in Hz of length `n`
    couplings : array-like
        an `n` x `n` array of coupling constants in Hz

    Returns
    -------
    ndarray
        a 2-D array for the spin Hamiltonian
    """
    nspins = len(freqlist)

    # Define Pauli matrices
    sigma_x = np.matrix([[0, 1 / 2], [1 / 2, 0]])
    sigma_y = np.matrix([[0, -1j / 2], [1j / 2, 0]])
    sigma_z = np.matrix([[1 / 2, 0], [0, -1 / 2]])
    unit = np.matrix([[1, 0], [0, 1]])

    # The following empty arrays will be used to store the
    # Cartesian spin operators.
    Lx = np.empty((1, nspins), dtype='object')
    Ly = np.empty((1, nspins), dtype='object')
    Lz = np.empty((1, nspins), dtype='object')

    for n in range(nspins):
        Lx[0, n] = 1
        Ly[0, n] = 1
        Lz[0, n] = 1
        for k in range(nspins):
            if k == n:  # Diagonal element
                Lx[0, n] = np.kron(Lx[0, n], sigma_x)
                Ly[0, n] = np.kron(Ly[0, n], sigma_y)
                Lz[0, n] = np.kron(Lz[0, n], sigma_z)
            else:  # Off-diagonal element
                Lx[0, n] = np.kron(Lx[0, n], unit)
                Ly[0, n] = np.kron(Ly[0, n], unit)
                Lz[0, n] = np.kron(Lz[0, n], unit)

    Lcol = np.vstack((Lx, Ly, Lz)).real
    Lrow = Lcol.T  # As opposed to sparse version of code, this works!
    Lproduct = np.dot(Lrow, Lcol)

    # Hamiltonian operator
    H = np.zeros((2 ** nspins, 2 ** nspins))

    # Add Zeeman interactions:
    for n in range(nspins):
        H = H + freqlist[n] * Lz[0, n]

    # Scalar couplings

    # Testing with MATLAB discovered J must be /2.
    # Believe it is related to the fact that in the SpinDynamics.org simulation
    # freqs are *2pi, but Js by pi only.
    scalars = 0.5 * couplings
    scalars = np.multiply(scalars, Lproduct)
    for n in range(nspins):
        for k in range(nspins):
            H += scalars[n, k].real

    return H


standard_3 = hamiltonian_original(*spin3())
standard_8 = hamiltonian_original(*spin8())
