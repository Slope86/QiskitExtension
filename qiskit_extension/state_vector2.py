"""Module to store, manipulate and visualize quantum state vector."""

from __future__ import annotations

import copy
import itertools
import re
from typing import Iterable, List, Tuple

import numpy as np
from IPython.display import Latex
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector
from scipy import stats

from qiskit_extension import latex_drawer
from qiskit_extension.ket import Ket


class StateVector2(Statevector):
    """An extended class of Statevector from Qiskit:
    https://qiskit.org/documentation/stubs/qiskit.quantum_info.Statevector.html

    Args:
        data (np.ndarray | list | Statevector | QuantumCircuit | Instruction):
            Data from which the statevector can be constructed. This can be either a complex
            vector, another statevector, a ``QuantumCircuit`` or ``Instruction``.
            If the data is a circuit or instruction, the statevector is constructed by assuming that
            all qubits are initialized to the zero state.
        dims (int | tuple | list, optional): The subsystem dimension of the state.
    """

    def __init__(self, data, dims: int | Iterable[int] | None = None) -> None:
        if not isinstance(data, np.ndarray | list | Statevector | QuantumCircuit | Instruction):
            raise QiskitError("Input only accepts np.ndarray | list | Statevector | QuantumCircuit | Instruction")

        # REVERSE the order of qubits to fit textbook notation
        if isinstance(data, QuantumCircuit):
            data = data.reverse_bits()
        elif isinstance(data, Instruction):
            data = Statevector(data).reverse_qargs()

        self._data: NDArray[np.complex128]
        super().__init__(data, dims)
        self._num_of_qubit = int(np.log2(len(self._data)))
        self._basis = ["z"] * self._num_of_qubit

    @classmethod
    def __init__with_basis(
        cls, data, dims: int | Iterable[int] | None = None, basis: List[str] | str = []
    ) -> StateVector2:
        """Initialize statevector with basis."""
        state_vector = cls(data, dims)
        state_vector.basis = list(basis) + ["z"] * (state_vector._num_of_qubit - len(basis))
        return state_vector

    def __repr__(self) -> str:
        """Return the official string representation of a Statevector."""
        vector_str = np.array2string(self.to_matrix(), separator=", ")
        return f"Statevector:\n{vector_str}"

    @property
    def data(self) -> NDArray[np.complex128]:
        return self._data.copy()

    @property
    def basis(self) -> List[str]:
        return self._basis.copy()

    @data.setter
    def data(self, data: NDArray[np.complex128]):
        self._data = data.copy()

    @basis.setter
    def basis(self, basis: List[str]):
        self._basis = basis.copy()

    def copy(self) -> StateVector2:
        """Return a copy of the current statevector."""
        return copy.deepcopy(self)

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return matrix form of statevector"""
        clone = self.copy()
        clone._basis_convert("z" * self._num_of_qubit)
        vector = clone._data
        matrix = vector[np.newaxis].T  # type: ignore
        return matrix

    def entropy(self, state: StateVector2 | Statevector | None = None) -> float:
        """Return entropy of input statevector."""
        if state is None:
            state = self
        return stats.entropy(state.probabilities(), base=2)  # type: ignore

    def evolve(self, circ: QuantumCircuit | NDArray, qargs: List[int] | None = None) -> StateVector2:
        """Evolve self(statevector) by a quantum circuit or matrix.
        return the evolved state as a new statevector. (self's state will not be changed)

        Args:
            circ (QuantumCircuit): The circ to evolve by.
            qargs (List[int], optional): A list of subsystem positions to apply the operator on.

        Returns:
            StateVector2: The new evolved statevector.
        """
        clone = self.copy()
        original_basis = clone.basis
        clone._basis_convert(["z"] * self._num_of_qubit)
        new_state = clone._evolve(circ, qargs)
        new_state._basis_convert(original_basis)
        return new_state

    def _evolve(self, circ: QuantumCircuit | NDArray, qargs: List[int] | None = None) -> StateVector2:
        """Evolve self(z-basis statevector) by a quantum circuit or matrix.
        return the evolved state as a new statevector. (self's state will not be changed)

        Args:
            circ (QuantumCircuit): The object to evolve by.
            qargs (List[int], optional): A list of subsystem positions to apply the operator on.

        Returns:
            StateVector2: The new evolved statevector.
        """
        if not isinstance(circ, QuantumCircuit | np.ndarray):
            raise QiskitError("Input is not a QuantumCircuit.")
        if isinstance(circ, QuantumCircuit):
            circ = circ.reverse_bits()  # REVERSE the order of qubits to fit textbook notation
        evolve_data = super().evolve(circ, qargs).data
        return StateVector2.__init__with_basis(data=evolve_data, basis=self.basis)

    def measure(
        self, measure: List[int] | int | str, basis: List[str] | str | None = None, shot=100
    ) -> List[StateVector2]:
        """measure statevector

        Args:
            measure (List[int] | int | str): which qubits to measure
            basis (List[str] | str | None, optional): measure in what basis. Defaults to current stored basis.
            shot (int, optional): number of shots. Defaults to 100.

        Returns:
            List[StateVector2]: list of statevector after measurement
        """
        states = self._measure(measure, basis, shot)[1]
        for state in states:
            if state is not None:
                state._basis_convert(self.basis)
        return states

    def _measure(
        self, measure: List[int] | int | str, basis: List[str] | str | None = None, shot=100
    ) -> Tuple[List[StateVector2], List[StateVector2]]:
        """measure statevector

        Args:
            measure (List[int] | int | str): which qubits to measure
            basis (List[str] | str, optional): measure in what basis. Defaults to current stored basis.
            shot (int, optional): number of shots. Defaults to 100.

        Returns:
            Tuple[List[StateVector2], List[StateVector2]]: list of measured statevector,
                list of remained statevector after measurement
        """
        clone = self.copy()

        if isinstance(measure, str):
            measure = [int(char) for char in measure]
        elif isinstance(measure, int):
            measure = [measure]
        if basis is not None:
            convert_basis = clone.basis
            for i in range(len(basis)):
                convert_basis[measure[i]] = basis[i]
            clone._basis_convert(convert_basis)

        measure_state_list = [None] * 2 ** len(measure)  # crate empty list for saving measure state
        remain_state_list = [None] * 2 ** len(measure)  # crate empty list for saving remain state
        for _ in range(shot):
            measure_ket: str
            remain_state: Statevector
            # REVERSE measure bit number to fit qiskit statevector measure notation
            reverse_measure = [clone._num_of_qubit - 1 - i for i in measure]
            measure_ket, remain_state = Statevector(clone.data).measure(qargs=reverse_measure)  # type: ignore
            measure_ket = measure_ket[::-1]  # REVERSE the order of qubits to fit textbook notation
            if measure_state_list[int(measure_ket, 2)] is None:
                basis = clone.basis
                measure_basis = ""
                for i in measure:
                    measure_basis += basis[i]

                basis_convert_measure_ket = ""
                for b, k in zip(measure_basis, measure_ket):
                    match b:
                        case "z":
                            basis_convert_measure_ket += Ket.z1 if int(k) else Ket.z0
                        case "x":
                            basis_convert_measure_ket += Ket.x1 if int(k) else Ket.x0
                        case "y":
                            basis_convert_measure_ket += Ket.y1 if int(k) else Ket.y0

                measure_state_list[int(measure_ket, 2)] = clone._from_label(basis_convert_measure_ket, to_z_basis=False)
                remain_state_list[int(measure_ket, 2)] = StateVector2.__init__with_basis(
                    data=remain_state.data, basis=clone.basis
                )
        return (measure_state_list, remain_state_list)

    def show_matrix(self):
        """show the matrix form of statevector in LaTeX"""
        return latex_drawer.matrix_to_latex(self.to_matrix())

    def show_state(
        self,
        basis: List[str] | str | None = None,
        hide: List[int] | str = [],
        output_length: int = 2,
        output: str = "latex",
    ):
        """visualize statevector

        Args:
            basis (List[str] | str, optional): convert statevector's basis to input basis.
                Defaults to skip basis convert.
            hide (List[int] | str, optional): hide qubits. Default to show all qubits.
            output_length (int, optional): 2^output_length = number of terms in each line. Defaults to 2(= 4 terms/line)
            output (str, optional): visualization method. Defaults to "latex".

        Returns:
            matplotlib.Figure or str or TextMatrix or IPython.display.Latex: visualization output
        """
        clone = self.copy()
        if basis is not None:
            clone._basis_convert(basis)

        match output:
            case "text":
                return print(clone)
            case "latex":
                return latex_drawer.state_to_latex(clone, hide, output_length)
            case "latex_source":
                return latex_drawer.state_to_latex(clone, hide, output_length, source=True)
            case _:
                return clone.draw(output=output)

    def show_measure(
        self,
        measure: List[int] | int | str,
        basis: List[str] | str | None = None,
        hide: List[int] | str = [],
        output_length: int = 2,
        remaining_basis: List[str] | str | None = None,
    ) -> Latex:
        """visualize measure result

        Args:
            measure (List[int] | int | str): qubits to measure
            basis (List[int] | str, optional): measure in what basis. Defaults to z basis.
            hide (List[int] | str, optional): hide qubits. Default to show all qubits.
            output_length (int, optional): 2^output_length = number of terms in each line. Defaults to 2(= 4 terms/line)
            remaining_basis (List[str] | str, optional): change statevector's basis to input basis before measure.

        Returns:
            Latex: visualization output
        """
        if isinstance(hide, str):
            hide = [int(char) for char in hide]
        elif isinstance(hide, int):
            hide = [hide]

        if isinstance(measure, str):
            measure = [int(char) for char in measure]
        elif isinstance(measure, int):
            measure = [measure]

        clone = self.copy()
        if remaining_basis is not None:
            remaining_basis = list(remaining_basis) + ["-"] * (self._num_of_qubit - len(remaining_basis))
            if basis is None:
                basis = ["z"] * len(measure)
            for i, b in zip(measure, basis):
                remaining_basis[i] = b
            clone._basis_convert(remaining_basis)
        result = clone._measure(measure=measure, basis=basis, shot=4 * 2 ** len(measure))
        return latex_drawer.measure_result_to_latex(result, measure + hide, output_length)  # type: ignore

    @classmethod
    def from_int(cls, i: int, dims: int | Iterable[int]) -> StateVector2:
        """Return a computational basis statevector.

        Args:
            i (int): the basis state element.
            dims (int | Iterable[int]): The subsystem dimensions of the statevector

        Returns:
            StateVector2: The statevector object.
        """
        state = super().from_int(i, dims)
        return cls(state.data, dims)

    @classmethod
    def from_instruction(cls, instruction):
        """Return the output statevector of an instruction.

        The statevector is initialized in the state :math:`|{0,\\ldots,0}\\rangle` of the
        same number of qubits as the input instruction or circuit, evolved
        by the input instruction, and the output statevector returned.

        Args:
            instruction (qiskit.circuit.Instruction or QuantumCircuit): instruction or circuit

        Returns:
            Statevector: The final statevector.

        Raises:
            QiskitError: if the instruction contains invalid instructions for
                         the statevector simulation.
        """
        if not isinstance(instruction, QuantumCircuit | Instruction):
            raise QiskitError("Input is not a valid instruction or circuit.")
        return cls(instruction)

    @classmethod
    def from_label(cls, *args: str | Tuple[complex, str]) -> StateVector2:
        """Create a state vector from input coefficient and label string.

        Example:
            from_label("0", "1") = (|0> + |1>)/√2,
            from_label("00", "01", "10", "11") = (|00> + |01> + |10> + |11>)/2 = |++>,
            from_label( (2**0.5,"0") , "+" , (-1,"-") ) = ( √2|0> + |+> - |-> )/2 = |+>

        Args:
            args (str | Tuple[complex, str]): The input label string or tuple of coefficient and label string.

        Returns:
            StateVector2: The statevector object.

        Raises:
            QiskitError: if labels contain invalid characters or labels have different number of qubits.
        """
        coeffs: List[complex] = []
        labels: List[str] = []
        for i, arg in enumerate(args):
            if isinstance(arg, tuple):
                coeffs.append(arg[0])
                labels.append(arg[1])
            else:
                coeffs.append(1.0)
                labels.append(arg)
            if not Ket.check_valid(labels[i]):
                raise QiskitError("Invalid label string.")
            if len(labels[0]) != len(labels[i]):
                raise QiskitError("Each label's number of qubits must be the same.")

        state: StateVector2 = cls._from_label(labels[0]) * coeffs[0]
        for coeff, label in zip(coeffs[1:], labels[1:]):
            state += cls._from_label(label) * coeff

        if type(state) is not StateVector2:
            raise QiskitError("Unexpected error.")
        state /= super(type(state), state).trace() ** 0.5
        return state

    @classmethod
    def _from_label(cls, label: str, to_z_basis: bool = True) -> StateVector2:
        if to_z_basis:
            label = label.replace(Ket.z0, "0")
            label = label.replace(Ket.x0, "+")
            label = label.replace(Ket.y0, "r")
            label = label.replace(Ket.z1, "1")
            label = label.replace(Ket.x1, "-")
            label = label.replace(Ket.y1, "l")
            state = super().from_label(label)
            return cls(state.data)

        basis = ""
        for ket in label:
            match ket:
                case Ket.z0 | Ket.z1:
                    basis += "z"
                case Ket.x0 | Ket.x1:
                    basis += "x"
                case Ket.y0 | Ket.y1:
                    basis += "y"

        label = label.replace(Ket.z0, "0")
        label = label.replace(Ket.x0, "0")
        label = label.replace(Ket.y0, "0")
        label = label.replace(Ket.z1, "1")
        label = label.replace(Ket.x1, "1")
        label = label.replace(Ket.y1, "1")
        state = super().from_label(label)
        return cls.__init__with_basis(state.data, basis=basis)

    def expand(self, other):
        """Return the tensor product state other ⊗ self.

        Args:
            other (Statevector2): a quantum state object.

        Returns:
            Statevector: the tensor product state other ⊗ self.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        clone = self.copy()
        clone._basis_convert("z" * self._num_of_qubit)
        other_clone = other.copy()
        other_clone._basis_convert("z" * other._num_of_qubit)
        state_expand = super(type(other_clone), other_clone).expand(clone)
        return StateVector2(state_expand.data)

    def _basis_convert(
        self,
        basis: List[str] | str = [],
        algorithm: str = "global",
    ) -> None:
        """Convert basis of statevector

        Args:
            basis (List[str] | str, optional):new basis. Defaults to auto choose basis with minimum entropy.
            algorithm (str, optional): if don't specify which basis to convert, (basis = "-")
                convert basis to basis with "local" or "global" minimum entropy. Defaults to "global".
        """
        # check if input is valid
        target_basis = list(basis) + ["-"] * (self._num_of_qubit - len(basis))
        del basis
        if re.match(R"^[\-xyz]+$", "".join(target_basis)) is None:
            raise QiskitError("Invalid basis.")

        # convert basis using QuantumCircuit
        auto_basis_index = []
        circ_convert = QuantumCircuit(self._num_of_qubit)
        for i in range(self._num_of_qubit):
            if target_basis[i] == self._basis[i]:
                continue
            if target_basis[i] != "-":
                self._xyz_convert_circ(self.basis[i], target_basis[i], circ_convert, i)
                self._basis[i] = target_basis[i]
            else:
                auto_basis_index.append(i)
        self._data = self._evolve(circ_convert)._data
        if not auto_basis_index:
            return

        # if user don't specify which basis to convert, convert basis to basis with minimum entropy
        match algorithm:
            case "global":
                if len(auto_basis_index) > 8:
                    print(
                        "Warning: global minimum entropy basis convert with more then 8 qubits might take a long time."
                    )
                new_basis = self._global_min_entropy_basis(auto_basis_index)
            case "local":
                new_basis = self._local_min_entropy_basis(auto_basis_index)
            case _:
                raise QiskitError("Invalid min_entropy_basis_find_method.")
        self._basis_convert(new_basis)

    def _global_min_entropy_basis(self, auto_basis_index: List[int]) -> List[str]:
        """find basis with global minimum entropy

        Args:
            auto_basis_index (List[int]): index of auto-choose-basis

        Returns:
            List[str]: basis with global minimum entropy
        """
        num_of_auto_basis = len(auto_basis_index)
        min_entropy = float("inf")
        min_basis = self.basis
        try_basis = self.basis
        for basis in itertools.product(["z", "x", "y"], repeat=num_of_auto_basis):  # type: ignore
            for i in range(num_of_auto_basis):
                try_basis[auto_basis_index[i]] = basis[i]
            try_state = self.copy()
            try_state._basis_convert(try_basis)
            if (entropy := try_state.entropy()) < min_entropy:
                min_entropy = entropy
                min_basis = try_basis.copy()
        return min_basis

    def _local_min_entropy_basis(self, auto_basis_index: List[int]) -> List[str]:
        """find basis with local minimum entropy

        Args:
            auto_basis_index (List[int]): index of auto-choose-basis

        Returns:
            List[str]: basis with local minimum entropy
        """
        # Step 1: Change all auto-choose-basis to y, e.g. [-, -, -, -] -> [z, z, z, z], calculate entropy
        # Step 2,3: Same as Step 1, but with x-basis and y-basis
        # Step 4: from Step 1 to 3, choose the basis with minimum entropy.
        clone_state = self.copy()

        min_entropy = float("inf")
        min_basis = clone_state.basis
        for basis in ["z", "x", "y"]:
            try_state = clone_state.copy()
            try_basis = try_state.basis
            for i in auto_basis_index:
                try_basis[i] = basis
            try_state._basis_convert(try_basis)
            if (entropy := try_state.entropy()) < min_entropy:
                min_entropy = entropy
                min_basis = try_basis
        clone_state._basis_convert(min_basis)

        # Step 1: Change the first auto-choose-basis to y, e.g. [-, -, -, -] -> [y, -, -, -], calculate entropy,
        # Step 2,3: Same as Step 1, but with x-basis and z-basis
        # Step 4: from Step 1 to 3, choose the basis with minimum entropy.
        # Step 5: Repeat Step 1 to 4 for the second auto-choose-basis, and so on. (greedy)
        # e.g. [-, -, -, -] -> [x, -, -, -] -> [x, z, -, -] -> [x, z, y, -] -> [x, z, y, z]
        min_entropy = float("inf")
        min_basis = clone_state.basis
        for i in auto_basis_index:
            try_basis = clone_state.basis
            for basis in ["y", "x", "z"]:
                try_basis[i] = basis
                try_state = clone_state.copy()
                try_state._basis_convert(try_basis)
                if (entropy_tmp := try_state.entropy()) < min_entropy:
                    min_entropy = entropy_tmp
                    min_basis[i] = basis
            clone_state._basis_convert(min_basis)
        return min_basis

    @staticmethod
    def _xyz_convert_circ(basis: str, new_basis: str, circ: QuantumCircuit, qubit: int) -> None:
        """Add the corresponding gate that converts different basis

        Args:
            basis (str): original basis
            new_basis (str): new basis
            circ (QuantumCircuit): quantum circuit to convert basis

        Returns:
            QuantumCircuit: quantum circuit to convert basis
        """
        if basis == new_basis:
            return
        basis += new_basis
        match basis:
            case "zx" | "xz":
                circ.h(qubit)
            case "zy":
                circ.sdg(qubit)
                circ.h(qubit)
            case "yz":
                circ.h(qubit)
                circ.s(qubit)
            case "xy":
                circ.h(qubit)
                circ.sdg(qubit)
                circ.h(qubit)
            case "yx":
                circ.h(qubit)
                circ.s(qubit)
                circ.h(qubit)
        return
