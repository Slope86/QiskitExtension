
# Qiskit StateVector extension

A simple Qiskit StateVector extension with more visualization function:

* Modifies the original StateVector visualization, adds line break and alignment.
* StateVector visualization with x, y, z basis.
* StateVector measurement with x, y, z basis, and visualizes all the possible measurement result.
* Visualizes the vector of a StateVector and the unitary matrix of a QuantumCircuit as a visually pleasing matrix in LaTeX format.
* Uses textbook qubit order. (for instance, if the first qubit is in state |0> and second is in state |1>, their joint state would be visualize as |01>)

## Installation

* Option 1. install from PyPI

    ```bash
    pip install QiskitExtension
    ```

    Than install MikTeX from [here](https://miktex.org/) for LaTeX visualization.

* Option 2. install from source

    ```bash
    git clone --depth 1 https://github.com/Slope86/QiskitExtension
    cd QiskitExtension
    pip install .
    ```

    Than install MikTeX from [here](https://miktex.org/) for LaTeX visualization.

## Configuration

The configuration file is located at `~/qiskit_extension/config/config.ini`

```ini
; This section sets up the notation for the StateVector(affect the visualization result and the constructor funcition from_label() ).
; The default notation uses |j> to represent |-i>. 
; You can change the notation to other character if necessary. (only accept single character.)
[ket]
z0 = 0
z1 = 1
x0 = +
x1 = -
y0 = i
y1 = j
```

## Usage

[中文示範](https://github.com/Slope86/QiskitExtension/blob/master/examples/0.%20%E5%9F%BA%E6%9C%AC%E6%93%8D%E4%BD%9C.ipynb) / [English demo](https://github.com/Slope86/QiskitExtension/blob/master/examples/eng%20(beta)/0.%20Basic%20operation.ipynb)

## Examples

Creating an EPR pair with circuit:

```python
from qiskit_extension.quantum_circuit2 import QuantumCircuit2 as qc2
from qiskit_extension.state_vector2 import StateVector2 as sv2

# the circuit for creating EPR pair
circuit_EPR = qc2(2)
circuit_EPR.h(0)
circuit_EPR.cx(0, 1)

# create a ground state
state00 = sv2.from_label('00')

# evolve the state with the circuit
state_EPR = state00.evolve(circuit_EPR)

# visualize the state
state_EPR.show_state() #|00> + |11>
```

Creating an EPR pair from label:
  
```python
from qiskit_extension.state_vector2 import StateVector2 as sv2

# create an EPR pair from label
state_EPR = sv2.from_label("00","11")

# visualize the state
state_EPR.show_state() #1/√2(|00> + |11>)
```

Visualize the state with different basis:

```python
from qiskit_extension.state_vector2 import StateVector2 as sv2

# create an EPR pair from label
state_EPR = sv2.from_label("00","11")

# visualize the state
state_EPR.show_state()           #1/√2(|00> + |11>)
state_EPR.show_state(basis='xx') #1/√2(|++> + |-->)
state_EPR.show_state(basis='yy') #1/√2(|ij> + |ji>)
```

More examples can be found in the [examples](https://github.com/Slope86/QiskitExtension/tree/master/examples) folder.

## Requirement

[MiKTeX](https://miktex.org/) (for LaTeX visualization)  
Python >= 3.10  
qiskit[visualization] == 1.0.0  
qiskit-aer == 0.13.3

## License

This QiskitExtension project is open source under the MIT license.
However, the extensions that are installed separately are not part of the QiskitExtension project.
They all have their own licenses!
