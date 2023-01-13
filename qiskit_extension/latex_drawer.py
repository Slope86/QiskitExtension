"""Module to convert various data types to LaTeX representation.
IPython.display.Latex(KaTeX) Supported Functions: https://katex.org/docs/supported.html
"""

from __future__ import annotations

import math
import typing
from typing import List, Tuple

import numpy as np
import sympy
from IPython.display import Latex
from numpy.typing import NDArray

from qiskit_extension.ket import Ket
from qiskit_extension.utils.find_nth_substring import find_nth_substring
from qiskit_extension.utils.float_gcd import float_gcd

if typing.TYPE_CHECKING:
    from qiskit_extension.state_vector2 import StateVector2


def matrix_to_latex(matrix: NDArray[np.float128 | np.complex128]) -> Latex:
    """Convert matrix to latex representation

    Args:
        matrix (NDArray[np.float128]): The matrix to be converted

    Returns:
        Latex: LaTeX representation of the matrix
    """
    prefix = R"$\begin{bmatrix}"
    suffix = R"\end{bmatrix}$"

    # If input is a 1d array, convert it to a column matrix
    if len(matrix.shape) == 1:
        matrix = matrix[np.newaxis].T

    # Extract the common factor from the matrix
    gcd = float_gcd(*np.absolute(matrix.flatten()))
    if not (math.isclose(gcd, 1) or math.isclose(gcd, 0)):
        matrix = matrix / gcd  # type: ignore
        pretty_gcd = _num_to_latex_ket(gcd)
        prefix = prefix[:1] + pretty_gcd + prefix[1:]

    # Convert the matrix to latex code
    latex_list: List[str] = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Convert the value to latex code
            pretty_valve = _num_to_latex_ket(matrix[i][j])
            latex_list.append(pretty_valve)
            # Add line up symbol
            if j != matrix.shape[1] - 1:
                latex_list.append("&")
        # Add line break symbol at the end of each row
        latex_list.append(R"\\[6pt]")

    latex_code = prefix + "".join(latex_list) + suffix
    return Latex(latex_code)


def state_to_latex(state: StateVector2, hide: List[int] | str = [], output_length: int = 2) -> Latex:
    """Convert state vector to latex representation

    Args:
        state (StateVector2):state vector
        hide (List[int] | str, optional): hide qubits. Default to show all qubits.
        output_length (int, optional): 2^output_length = number of terms in each line. Defaults to 2(= 4 terms/line).

    Returns:
        Latex: LaTeX representation of the state vector
    """
    prefix = R"$\begin{alignedat}{" + f"{2**(output_length+1)+1}" + R"}&\; \;&\;"
    suffix = R"\end{alignedat}$"
    latex_code = _state_to_latex_ket(state, hide)
    latex_code = _latex_line_break(latex_code, output_length)
    latex_code = prefix + latex_code + suffix
    latex_code = latex_code.replace(R"\;&\;-", R"-&\;")
    return Latex(latex_code)


def measure_result_to_latex(
    result: Tuple[List[StateVector2], List[StateVector2]],
    hide: List[int] | str = [],
    output_length: int = 2,
) -> Latex:
    """Convert measurement result to latex representation

    Args:
        result (Tuple[StateVector2, StateVector2]): measurement result
        hide (List[int] | str, optional): hide qubits. Default show all qubits.
        output_length (int, optional): 2^output_length = number of terms in each line. Defaults to 2(= 4 terms/line).

    Returns:
        Latex: LaTeX representation of the measurement result
    """
    prefix = R"$\begin{alignedat}{" + f"{2**(output_length+1)+1}" + "}"
    suffix = R"\end{alignedat}$"

    measure_state_list, remain_state_list = result
    latex_list: List[str] = []
    for measure_state, remain_state in zip(measure_state_list, remain_state_list):
        if measure_state is not None:
            latex_list.append(_state_to_latex_ket(state=measure_state)[:-1])
            latex_list.append(R":&\; \;&\;")
            tmp_str = _state_to_latex_ket(state=remain_state, hide=hide)
            latex_list.append(_latex_line_break(tmp_str, output_length))
            latex_list.append(R"\\\\")
    latex_code = prefix + "".join(latex_list) + suffix
    latex_code = latex_code.replace(R"\;&\;-", R"-&\;")
    return Latex(latex_code)


def _state_to_latex_ket(state: StateVector2, hide: List[int] | str = []) -> str:
    """Convert state vector to latex representation, modified from qiskit.visualization.state_visualization

    Args:
        data: State vector
        hide (List[int] | str, optional): hide qubits. Default show all qubits.

    Returns:
        str: String with LaTeX representation of the state vector
    """
    data = state.data
    basis = state.basis
    num = state._num_of_qubit
    if isinstance(hide, str):
        hide = [int(char) for char in hide]
    for i in hide:
        basis[i] = "hide"

    def ket_name(i):
        ket = bin(i)[2:].zfill(num)
        # ket = ket[::-1]  # REVERSE the order of qubits to fit textbook notation
        new_ket = ""
        for b, k in zip(basis, ket):
            match b:
                case "hide":
                    continue
                case "z":
                    new_ket += Ket.z1 if int(k) else Ket.z0
                case "x":
                    new_ket += Ket.x1 if int(k) else Ket.x0
                case "y":
                    new_ket += Ket.y1 if int(k) else Ket.y0
        return new_ket

    data = np.around(data, 15)
    nonzero_indices = np.where(data != 0)[0].tolist()
    latex_terms = _coeffs_to_latex_terms(data[nonzero_indices], decimals=15)

    latex_list: List[str] = []
    for idx, ket_idx in enumerate(nonzero_indices):
        if ket_idx is None:
            latex_list.append(R" + \ldots ")
        else:
            term = latex_terms[idx]
            ket = R"\texttt{" + ket_name(ket_idx) + "}"
            latex_list.append(Rf"{term}|{ket}\rangle &")

    return "".join(latex_list)


def _coeffs_to_latex_terms(coeffs: NDArray[np.complex128], decimals: int = 10) -> List[str]:
    """Convert a list of coefficients to latex formatted terms.

    The first non-zero term is treated differently. For this term a leading + is suppressed.

    Args:
        coeffs: List of coefficients to format
        decimals: Number of decimal places to round to (default: 10).
    Returns:
        List of formatted terms
    """
    first_term = True
    terms = []
    for coeff in coeffs:
        term = _coeff_to_latex_ket(coeff, first_term, decimals)
        if term is not None:
            first_term = False
        terms.append(term)
    return terms


def _coeff_to_latex_ket(raw_value: complex, first_coeff: bool, decimals: int = 10) -> str | None:
    """Convert a complex coefficient to latex code suitable for a ket expression.

    Args:
        raw_value: Value to convert
        first_coeff: If True then generate latex code for the first term in an expression
        decimals: Number of decimal places to round to (default: 10).
    Returns:
        String with latex code or None if no term is required
    """
    # Round to the specified number of decimals
    raw_value = np.around(raw_value, decimals=decimals)

    # If the value is zero then return None
    if np.abs(raw_value) == 0:
        return None

    # If the value has both real and imaginary parts, and the real part is negative, then extract the minus sign.
    # for example: +(-0.5+0.5j) -> -(0.5+0.5j)
    real_value = raw_value.real
    imag_value = raw_value.imag
    two_term_sign = "+"
    if np.sign(real_value) == -1 and imag_value != 0:
        two_term_sign = "-"
        raw_value = -raw_value

    # Convert to a sympy expression, then to latex
    value = sympy.nsimplify(raw_value, constants=(sympy.pi,), rational=False)
    latex_element = sympy.latex(value, full_prec=False)

    # Check if the value has more than one term
    two_term = real_value != 0 and imag_value != 0
    if isinstance(value, sympy.core.Add):
        # can happen for expressions like 1 + sqrt(2)
        two_term = True

    # If the value is 1 or -1 then suppress the coefficient
    if latex_element == "1":
        # If this is the first coefficient then suppress the leading + sign
        if first_coeff:
            return ""
        return "+"
    if latex_element == "-1":
        return "-"

    # If the value has more than one term, wrap it in parentheses
    if two_term:
        # If this is the first coefficient then suppress the leading + sign
        if first_coeff and two_term_sign == "+":
            return f"({latex_element})"
        return f"{two_term_sign}({latex_element})"

    # If this is not the first coefficient and the value is positive then add a leading + sign
    if not first_coeff and latex_element[0] != "-":
        return f"+{latex_element}"

    # Other normal case
    return latex_element


def _num_to_latex_ket(raw_value: complex, decimals: int = 15) -> str:
    """Convert a complex number to latex element.

    Args:
        raw_value: Value to convert
        decimals: Number of decimal places to round to (default: 15).

    Returns:
        str: latex element representing the value
    """
    raw_value = np.around(raw_value, decimals=decimals)
    value = sympy.nsimplify(raw_value, constants=(sympy.pi,), rational=False)
    return sympy.latex(value, full_prec=False)


def _latex_line_break(latex_code: str, output_length: int = 2) -> str:
    """Split latex string into several lines, so that each line has 2^output_length terms.

    Args:
        latex_code (str): latex source code
        output_length (int, optional): 2^output_length = number of terms in each line. Defaults to 2(= 4 terms/line).

    Returns:
        str: result latex code
    """
    latex_code = latex_code.replace("|", "&|")
    max_term = 2**output_length
    num_of_term = latex_code.count(R"\rangle")
    line_break = R"\\ &&"
    for i in range(max_term, num_of_term, max_term):
        new_line_index = find_nth_substring(latex_code, R"\rangle", i) + 7
        latex_code = latex_code[:new_line_index] + line_break + latex_code[new_line_index + 2 :]

    latex_code = latex_code.replace("&+", R"\;+&\;")
    latex_code = latex_code.replace("&-", R"\;-&\;")
    return latex_code
