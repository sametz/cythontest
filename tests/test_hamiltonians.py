import os
# from timeit import timeit

import numpy as np
import pytest
import send2trash

from simulation_data import spin8, spin11, spin3, fox
from speedtest.speedutils import timefn  # Decorator for timing functions
from speedtest.compare_hamiltonians import (kuprov_H, kuprov_cached, hamiltonian_slow, hamiltonian,
                                            spin_operators_unvectorized, spin_operators_vectorized,
                                            hamiltonian_unvectorized, hamiltonian_vectorized,
                                            hamiltonian_sparse)
from nmrtools.nmrmath import simsignals
from .prepare import standard_3, standard_8, standard_11  #standard_fox

SPIN_SYSTEM = spin11
STANDARD_H = standard_8

def cleanup():
    path = os.getcwd()
    spin8_operators = ['Lproduct8.npy', 'Lz8.npy']
    operators = [os.path.join(path, operator) for operator in spin8_operators]
    files_cleaned = []
    for operator in operators:
        if os.path.isfile(operator):
            files_cleaned.append(operator)
            send2trash.send2trash(os.path.basename(operator))
    return files_cleaned


@pytest.fixture
def cleanup_fixture():
    yield
    return cleanup()


def test_tmpdir(tmpdir):
    """Smoke test.

    From "Python Testing with pytest" by Brian Okken
    """
    # tmpdir already has a path name associated with it
    # join() extends the path to include a filename
    # the file is created when it's written to
    a_file = tmpdir.join('something.txt')

    # you can create directories
    a_sub_dir = tmpdir.mkdir('anything')

    # you can create files in directories (created when written)
    another_file = a_sub_dir.join('something_else.txt')

    # this write creates 'something.txt'
    a_file.write('contents may settle during shipping')

    # this write creates 'anything/something_else.txt'
    another_file.write('something different')

    # you can read the files as well
    assert a_file.read() == 'contents may settle during shipping'
    assert another_file.read() == 'something different'


def test_cleanup():
    # make a mess
    v, J = spin8()
    hamiltonian(v, J)
    # then clean it up
    files_cleaned = cleanup()
    print('FILES CLEANED:')
    print(files_cleaned)
    path = os.getcwd()
    spin8_operators = ['Lproduct8.npy', 'Lz8.npy']
    operators = [os.path.join(path, operator) for operator in spin8_operators]
    assert files_cleaned == operators
    for operator in operators:
        assert not os.path.isfile(operator), 'File left behind! ' + operator


def test_kuprov():
    v, J = SPIN_SYSTEM()
    test_H = kuprov_H(v, J)
    assert np.array_equal(test_H, STANDARD_H)


def test_hamiltonian_slow():
    v, J = SPIN_SYSTEM()
    test_H = hamiltonian_slow(v, J)
    assert np.array_equal(test_H, STANDARD_H)


def test_hamiltonian():
    v, J = SPIN_SYSTEM()
    test_H = hamiltonian(v, J)
    assert np.array_equal(test_H, STANDARD_H)


def test_hamiltonian_unvectorized():
    v, J = SPIN_SYSTEM()
    L = spin_operators_unvectorized(len(v))
    test_H = hamiltonian_unvectorized(v, J, L)
    assert np.array_equal(test_H, STANDARD_H)


def test_hamiltonian_vectorized():
    v, J = SPIN_SYSTEM()
    L = spin_operators_vectorized(len(v))
    test_H = hamiltonian_vectorized(v, J, L)
    assert np.array_equal(test_H, STANDARD_H)


@timefn
def kuprov_loop(v, J, n):
    for i in range(n):
        _ = kuprov_H(v, J)
    return _


@timefn
def kuprov_cached_loop(v, J, n):
    for i in range(n):
        _ = kuprov_cached(v, J)
    return _


@timefn
def slow_loop(v, J, n):
    for i in range(n):
        _ = hamiltonian_slow(v, J)
    return _


@timefn
def hamiltonian_loop(v, J, n):
    for i in range(n):
        _ = hamiltonian(v, J)
        # cleanup()
    return _


@timefn
def unvectorized_loop(v, J, n):
    for i in range(n):
        # L = spin_operators(len(v))
        _ = hamiltonian_unvectorized(v, J)
    return _


@timefn
def vectorized_loop(v, J, n):
    for i in range(n):
        # L = spin_operators(len(v))
        _ = hamiltonian_vectorized(v, J)
    return _


@timefn
def sparse_loop(v, J, n):
    for i in range(n):
        # L = spin_operators(len(v))
        _ = hamiltonian_sparse(v, J)
    return _


def test_all():
    n = 1
    v, J = SPIN_SYSTEM()
    # kuprov_loop(v, J, n)
    # kuprov_cached_loop(v, J, n)
    # slow_loop(v, J, n)
    hamiltonian_loop(v, J, n)
    # unvectorized_loop(v, J, n)
    # vectorized_loop(v, J, n)
    sparse_loop(v, J, n)
    assert True


def test_all_equal():
    v, J = SPIN_SYSTEM()
    h_funcs = [kuprov_cached, hamiltonian, hamiltonian_vectorized, hamiltonian_sparse]
    hamiltonians = [f(v, J) for f in h_funcs]
    for i in range(len(hamiltonians) - 1):
        try:
            assert np.array_equal(hamiltonians[i], hamiltonians[i + 1]), 'fail at ' + str(i)
            print('Passed: ', str(i), str(i + 1))
        except AssertionError:
            print('failure at ', str(i))
            print(hamiltonians[i])
            print(hamiltonians[i + 1])


def test_all_eigen():
    v, J = SPIN_SYSTEM()
    nspins = len(v)
    h_funcs = [kuprov_cached, hamiltonian, hamiltonian_vectorized, hamiltonian_sparse]
    hamiltonians = [f(v, J) for f in h_funcs]
    spectra = [simsignals(h, nspins) for h in hamiltonians]
    for i in range(len(spectra) - 1):
        try:
            np.testing.assert_allclose(spectra[i], spectra[i + 1]), 'fail at ' + str(i)
            print('Passed: ', str(i), str(i + 1))
        except AssertionError:
            print('failure at ', str(i))
        print(spectra[i])
        print(spectra[i + 1])

# if __name__ == '__main__':
#     from simulation_data import spin8
#     v, J = spin8()
#     L = spin_operators(8)
#     hamiltonians = [kuprov_H(v, J), hamiltonian_slow(v, J), hamiltonian(v, J), hamiltonian_unvectorized(v, J, L),
#                     hamiltonian_vectorized(v, J, L)]
#     for i in range(len(hamiltonians)-1):
#         assert np.array_equal(hamiltonians[i], hamiltonians[i+1])
