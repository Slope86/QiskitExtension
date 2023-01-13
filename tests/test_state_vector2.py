import math

import numpy as np
from numpy.typing import NDArray

from qiskit_extension.ket import Ket
from qiskit_extension.state_vector2 import StateVector2 as sv2

ABS_TOL = 1e-15  # absolute tolerance for close comparisons

Z0 = Ket.z0
Z1 = Ket.z1
X0 = Ket.x0
X1 = Ket.x1
Y0 = Ket.y0
Y1 = Ket.y1


def test_from_label():
    # |0+r>
    state = sv2._from_label(Z0 + X0 + Y0, to_z_basis=False)
    assert state._basis == ["z", "x", "y"]
    assert (
        state._data == [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
    ).all()

    # |1+l>
    state = sv2._from_label(Z1 + X1 + Y1, to_z_basis=False)
    assert state._basis == ["z", "x", "y"]
    assert (
        state._data == [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]
    ).all()

    # |0> - i|1> = |-i>
    state = sv2.from_label((1, Z0), (-1j, Z1))
    assert state._basis == ["z"]
    assert state.draw("latex_source") == super(sv2, sv2).from_label("l").draw("latex_source")


def test_basis_convert():
    # |010+-+rlr>
    state = sv2.from_label(Z0 + Z1 + Z0 + X0 + X1 + X0 + Y0 + Y1 + Y0)
    state._basis_convert("xyzxyzxyz")
    assert state._basis == ["x", "y", "z", "x", "y", "z", "x", "y", "z"]
    state._basis_convert(algorithm="local")
    assert state._basis == ["z", "z", "z", "x", "x", "x", "y", "y", "y"]


def test_entropy():
    # |000>
    state = sv2.from_label(Z0 * 3)
    assert math.isclose(state.entropy(), 0.0, abs_tol=ABS_TOL)

    # |0+0> + |0-0>
    state._basis_convert("-x-")
    assert math.isclose(state.entropy(), 1.0, abs_tol=ABS_TOL)

    # |r0r> + |r0l> + |l0r> + |l0l>
    state._basis_convert("y-y")
    assert math.isclose(state.entropy(), 2.0, abs_tol=ABS_TOL)

    # |000>
    state._basis_convert("")
    assert math.isclose(state.entropy(), 0.0, abs_tol=ABS_TOL)


def test_to_matrix():
    test_vector: NDArray
    expect_vector: NDArray

    # |000>
    test_vector = sv2.from_label(Z0 * 3).to_matrix().squeeze()
    expect_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    assert np.allclose(test_vector, expect_vector, atol=ABS_TOL)

    # |01>
    test_vector = sv2.from_label(Z0 + Z1).to_matrix().squeeze()
    expect_vector = np.array([0, 1, 0, 0])
    assert np.allclose(test_vector, expect_vector, atol=ABS_TOL)

    # |0+>
    test_vector = sv2.from_label(Z0 + X0).to_matrix().squeeze()
    expect_vector = np.array([1, 1, 0, 0]) / math.sqrt(2)
    assert np.allclose(test_vector, expect_vector, atol=ABS_TOL)

    # |0->
    test_vector = sv2.from_label(Z0 + X1).to_matrix().squeeze()
    expect_vector = np.array([1, -1, 0, 0]) / math.sqrt(2)
    assert np.allclose(test_vector, expect_vector, atol=ABS_TOL)

    # |1+>
    test_vector = sv2.from_label(Z1 + X0).to_matrix().squeeze()
    expect_vector = np.array([0, 0, 1, 1]) / math.sqrt(2)
    assert np.allclose(test_vector, expect_vector, atol=ABS_TOL)

    # |1->
    test_vector = sv2.from_label(Z1 + X1).to_matrix().squeeze()
    expect_vector = np.array([0, 0, 1, -1]) / math.sqrt(2)
    assert np.allclose(test_vector, expect_vector, atol=ABS_TOL)

    # |0++> + |1-->
    test_vector = sv2.from_label(Z0 + X0 + X0, Z1 + X1 + X1).to_matrix().squeeze()
    expect_vector = np.array([1, 1, 1, 1, 1, -1, -1, 1]) / math.sqrt(8)
    assert np.allclose(test_vector, expect_vector, atol=ABS_TOL)
