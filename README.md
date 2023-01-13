
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

## Requirement

Python >= 3.10\
qiskit[visualization] >= 0.22.3
