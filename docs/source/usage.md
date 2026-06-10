# How to Guide

## Installation

You can install the package directly from the source code. Download or clone the GitHub repository, navigate into the main directory, and install it using `pip`:

```bash
git clone [https://github.com/llorenc-11/SoDiferent.git](https://github.com/llorenc-11/SoDiferent.git)
cd SoDiferent
pip install .
```

---

## Code Showcase

The core function used for solving systems of Ordinary Differential Equations (ODEs) is `RungeKutta`. 

```python
from SoDiferent.ODEsolvers import RungeKutta
RungeKutta(function, t_min, t_max, initial_y, tolerance=1e-5, min_step=1e-4, extra_parameters=None, compile_to_C=False)
```
*(For a full breakdown of the arguments, please see the **Function Reference** section).*

### 1. Solving Systems of ODEs (Array-Like Structures)
`SoDiferent`  handles multi-dimensional systems. If you  pass a Python `list` or a`NumPy` array to the `initial_y` parameter, the engine automatically calculates the coupled equations.

Here is an example solving a 2D harmonic oscillator (a frictionless spring):

```python
import numpy as np

def harmonic_oscillator(t, y):
    dy0 = y[1]   # Velocity
    dy1 = -y[0]  # Acceleration
    return [dy0, dy1]


# We start at position = 1.0, velocity = 0.0
res_y, res_t = RungeKutta( function=harmonic_oscillator, t_min=0.0, t_max=10.0, initial_y=np.array([1.0, 0.0]) )

# res_y is a 2D array where column 0 is position and column 1 is velocity over time
# array([[ 1.        ,  0.        ],
#        [ 0.9681503 , -0.25036946],
#        [ 0.87485026, -0.48439318],
#        ...
#        [-0.8889587 ,  0.45798502],
#        [-0.7461909 ,  0.66573031]])
```

### 2. High-Speed Execution (C++ Compilation)
For computationally heavy systems, pure Python can be a bottleneck. By setting `compile_to_C=True`, the solver will use Numba to JIT-compile your Python function into raw machine code and give it directly into the C++ backend for better performance.
You can also pass static variables into your compiled function using the `extra_parameters` argument:

```python

def predator_prey(t, y, params):
    alpha, beta, delta, gamma = params
    
    d_prey = (alpha * y[0]) - (beta * y[0] * y[1])
    d_predator = (delta * y[0] * y[1]) - (gamma * y[1])
    return [d_prey, d_predator]

res_y, res_t = RungeKutta(function=predator_prey,t_min=0.0,t_max=50.0,initial_y=[10.0, 5.0], extra_parameters=(1.5, 0.1, 0.1, 1.5), compile_to_C=True)
# res_y is a 2D array where column 0 is Prey population and column 1 is Predator population
# array([[10.        ,  5.        ],
#        [10.00100006,  4.99975003],
#        [10.02497121,  4.99378634],
#        ...
#        [ 6.49630367,  6.93020426],
#        [ 8.01847082,  5.7404328 ],
#        [ 9.76126102,  5.06242972]])
```