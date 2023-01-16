"""Module to construct, simulate, and visualize quantum circuit"""

from __future__ import annotations

import itertools
import typing
from typing import List

import numpy as np
from IPython.display import Latex
from numpy.typing import NDArray
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, quantum_info, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

from qiskit_extension import latex_drawer
from qiskit_extension.state_vector2 import StateVector2

if typing.TYPE_CHECKING:
    import matplotlib.figure


class QuantumCircuit2(QuantumCircuit):
    """An extended class of QuantumCircuit from Qiskit:
    https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html

    Args:
        number_of_qubits (int, optional): Number of qubits. Defaults to 1.
    """

    def __init__(self, number_of_qubits: int = 1):
        # Use Aer's qasm_simulator
        self.simulator = AerSimulator()
        self.n = number_of_qubits
        q = QuantumRegister(self.n, "q")
        c = ClassicalRegister(self.n, "c")
        super().__init__(q, c)

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return matrix form of the quantum circuit"""
        reverse_qc = self.reverse_bits()  # REVERSE the order of qubits to fit textbook notation
        return quantum_info.Operator(reverse_qc).data

    def show_matrix(self) -> Latex:
        """Show the matrix form of the quantum circuit in LaTeX"""
        return latex_drawer.matrix_to_latex(self.to_matrix())

    def show_circ(self) -> Latex:
        """Show the circuit in LaTeX"""
        return self.draw(output="latex", idle_wires=False)  # type: ignore

    def show_measure_all(self) -> matplotlib.figure.Figure:
        """Measure every qubit at the end of the circuit, then plot the result

        Returns:
            matplotlib.figure.Figure: A histogram of the measurement result
        """
        for i in range(self.n):
            self.measure(i, i)

        # compile the circuit down to low-level QASM instructions
        # supported by the backend (not needed for simple circuits)
        compiled_circuit = transpile(self, self.simulator)

        # Execute the circuit on the qasm simulator
        job = self.simulator.run(compiled_circuit, shots=1000)

        # Grab results from the job
        result = job.result()

        # Returns counts
        counts = result.get_counts(self)
        print(f"\nTotal count : {counts}\n")

        # Plot a histogram
        return plot_histogram(counts)

    def get_state(self) -> StateVector2:
        """Initialize the state to ground state, then evolve the state by the quantum circuit.

        Returns:
            StateVector2: Final state of the quantum circuit
        """
        # Set the initial state of the simulator to the ground state using from_int
        state = StateVector2.from_int(i=0, dims=2**self.n)

        # Evolve the state by the quantum circuit
        state = state.evolve(self)
        return state

    def phase_shift(self, theta, qubit) -> None:
        """Phase shift gate

        Args:
            theta (_type_): phase shift angle
            qubit (_type_): which qubit to apply the gate
        """
        self.p(theta, qubit)
        self.x(qubit)
        self.p(theta, qubit)
        self.x(qubit)

    def h_all(self) -> None:
        """Perform H gate on every qubit"""
        for i in range(self.n):
            self.h(i)

    def cz_line_all(self) -> None:
        """Perform CZ gate on qubit(0,1), qubit(1,2), qubit(2,3) ... , qubit(n-1,n)"""
        for i in range(self.n - 1):
            self.cz(i, i + 1)

    def cz_star_all(self, center=0) -> None:
        """Perform CZ gate on (center qubit, every other qubit)

        Args:
            center (int, optional): Center qubit. Defaults to 0.
        """
        for i in range(self.n):
            if i == center:
                continue
            self.cz(center, i)

    def cz_complete(self) -> None:
        """Perform CZ gate on every possible combination of qubits"""
        for qubit_c, qubit_t in itertools.combinations(range(self.n), 2):
            self.cz(qubit_c, qubit_t)

    def z2epr_pair(self, qubit_c: List[int] | int, qubit_t: List[int] | int) -> None:
        """Perform Z to EPR pair transformation

        Args:
            qubit_c (List[int] | int): control qubit
            qubit_t (List[int] | int): target qubit
        """
        if isinstance(qubit_c, int) and isinstance(qubit_t, int):
            qubit_c, qubit_t = [qubit_c], [qubit_t]
        if isinstance(qubit_c, list) and isinstance(qubit_t, list):
            if len(qubit_c) != len(qubit_t):
                raise ValueError("qubit_c and qubit_t must have the same length")
            for c, t in zip(qubit_c, qubit_t):
                self.h(c)
                self.cx(c, t)
        else:
            raise TypeError("qubit_c and qubit_t must be List[int] or int")

    def epr2z_basis(self, qubit_c: List[int] | int, qubit_t: List[int] | int) -> None:
        """Perform EPR pair to Z basis transformation

        Args:
            qubit_c (List[int] | int): control qubit
            qubit_t (List[int] | int): target qubit
        """
        if isinstance(qubit_c, int) and isinstance(qubit_t, int):
            qubit_c, qubit_t = [qubit_c], [qubit_t]
        if isinstance(qubit_c, list) and isinstance(qubit_t, list):
            if len(qubit_c) != len(qubit_t):
                raise ValueError("qubit_c and qubit_t must have the same length")
            for c, t in zip(qubit_c, qubit_t):
                self.cx(c, t)
                self.h(c)
        else:
            raise TypeError("qubit_c and qubit_t must both be List[int] or both be int")
