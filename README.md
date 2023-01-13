
# Qiskit visualization extension

A simple qiskit extension with more visualization function.

## Installation

```bash
  git clone --depth 1 https://github.com/Slope86/QiskitExtension
  cd QiskitExtension
  pip install .
```

## Configuration

the configuration file for this extension is located at `~/qiskit_extension/config/config.ini`

```ini
; Setting up the ket notation for the state vector,
; e.g. |0>, |1>, |+>, |->, |i>, |j> (|j> = |-i>),
; can change to other notation if needed. (only accept char)
[ket]
z0 = 0
z1 = 1
x0 = +
x1 = -
y0 = i
y1 = j
```

## Usage/Examples

A simple example to create a EPR pair:

```python
  from qiskit_extension.quantum_circuit2 import QuantumCircuit2 as qc2
  from qiskit_extension.state_vector2 import StateVector2 as sv2

  # create the circuit for creating EPR pair
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

More examples can be found in the `examples` folder.

## Requirement

Python >= 3.10\
qiskit[visualization] >= 0.22.3

## License

The QiskitExtension project is open source under the MIT license.
However, the extensions that are installed separately are not part of the QiskitExtension project.
They all have their own licenses!
