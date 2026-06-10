
# SoDiferent

**A fast, C++ backed Ordinary Differential Equation (ODE) solver for Python.**

SoDiferent helps with hurdles of complex numerical integration methods so you can focus strictly on the mathematical details. Simply define your system of equations as a standard Python function, and the solver handles the rest.

## Why use SoDiferent?

* **Flexibility of Python:** Write your equations in pure Python, just like you always do.
* **C++ Backed:** The core mathematical solvers are written in C++ for maximum performance.
* **Automatic Compilation:** Using Numba, the package can automatically compile your Python functions directly into raw C-speed machine code before solving.

## Contributing
SoDiferent is distributed under the open source GNU AGPL v3 license. Its source code can be downloaded from [Github](https://github.com/llorenc-11/SoDiferent)
**Author: ** Leonard Lorenc





```{toctree}
:maxdepth: 2
:caption: Contents:

self

:caption User Guide:
usage
api