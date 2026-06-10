import numpy as np
import warnings
import numba as nb
from numba import types, cfunc, carray
import SoDiferent.NumericRk as numeric

def _compile_scalar(user_func, num_params):
    """Compiles a scalar function for C++"""
    compiled_func = nb.njit(user_func)

    c_sig = types.float64(types.float64, types.float64, types.CPointer(types.float64))

    if num_params > 0:
        @cfunc(c_sig, nopython=True)
        def c_wrapper(t, y, params_ptr):
            p_array = carray(params_ptr, (num_params,)) 
            return compiled_func(t, y, p_array)
        return c_wrapper
        
    else:
        @cfunc(c_sig, nopython=True)
        def c_wrapper(t, y, params_ptr):
            return compiled_func(t, y)
        return c_wrapper


def _compile_vectorized(user_func, num_equations, num_params):
    """Compiles an N-Dimensional system for C++"""
    compiled_func = nb.njit(user_func)
    
    c_sig = types.void(types.float64, types.CPointer(types.float64), 
    types.CPointer(types.float64), types.CPointer(types.float64))

    if num_params > 0:
        @cfunc(c_sig, nopython=True)
        def c_wrapper(t, y_ptr, dy_ptr, params_ptr):
            y_array = carray(y_ptr, (num_equations,))
            p_array = carray(params_ptr, (num_params,))
            res = compiled_func(t, y_array, p_array)
            for i in range(num_equations):
                dy_ptr[i] = res[i]
        return c_wrapper
        
    else:
        @cfunc(c_sig, nopython=True)
        def c_wrapper(t, y_ptr, dy_ptr, params_ptr):
            y_array = carray(y_ptr, (num_equations,))
            res = compiled_func(t, y_array)
            for i in range(num_equations):
                dy_ptr[i] = res[i]
        return c_wrapper


def RungeKutta(function, t_min, t_max, initial_y, tolerance=1e-5, min_step=1e-4, extra_parameters=None,compile_to_C = False):
    """
    Solves an initial value problem for a system of ordinary differential equations (ODEs) 
    using the explicit Dormand-Prince method from the adaptive Runge-Kutta family.
    This solver dynamically adjusts the step size to optimize computation time
    while ensuring that the error at each step remains below the specified tolerance.
    
    The method accepts a Python function written by the user, which is then used by the 
    solver written in C++. This is handled in two ways:
    
    - Python path: The solver uses the Python C-API to issue a call to the user's function. 
      It is executed by Python, and the result is passed back to C++.
    - Compiled path: If the user passes a function compiled by Numba, a raw C pointer, or 
      if the `compile_to_C` flag is set to True, the solver calls the compiled C code 
      directly. See the `compile_to_C` parameter description for more details.

    Parameters
    ----------
    function : callable, int, or numba.CFunc
        The derivative function to evaluate the system. 
        - If a Python callable: Must have the signature `f(t, y)` or `f(t, y, params)`. 
        - If an `int`: Treated as a raw native OS memory pointer to pre-compiled C/C++ code.
        - If a Numba `cfunc`: The compiled memory address will be extracted automatically.
        
    t_min : float
        The initial value of the independent variable (the variable with respect to which 
        the derivative is evaluated, for example: starting time).
        
    t_max : float
        The final value of the independent variable for integration.(e.g. end time)
        
    initial_y : float or ndarray
        The initial state of the system at `t_min`. If a float is provided, the solver 
        treats the system as a 1D scalar equation. If a 1D NumPy array or array-like 
        structure is provided, it is treated as a system of ODEs.
        
    tolerance : float, optional
        The maximum allowed local truncation error per step. Default is 1e-5.
        
    min_step : float, optional
        The minimum allowed absolute step size to prevent infinite loops in extremely 
        'stiff' regions. Default is 1e-4.
        
    extra_parameters : tuple, optional
        A tuple of additional arguments to be passed to the derivative function. 
        These arguments are passed as a single grouped parameter. For example, a signature 
        like `f(t, y, p1, p2, p3)` is not allowed; it must be `f(t, y, params)`, where 
        `params` is a tuple containing `(p1, p2, p3)`. If `compile_to_C` is set to True, 
        the types inside this tuple must be simple, numerical types (easily convertible 
        to float, e.g., int or bool).
        
    compile_to_C : bool, optional
        If True, attempts to compile the Python `function` into C machine code using Numba. 
        It first JIT-compiles the function (allowing features like NumPy arrays), and then 
        compiles a C-wrapper that allows the C++ solver to execute it directly. If 
        compilation fails due to Python features unsupported by C (e.g., dictionaries), 
        it emits a RuntimeWarning and safely falls back to the standard Python execution 
        engine. Default is False.

    Returns
    -------
    y_values : ndarray
        The computed solution values at each step (calculated by the solver at points 
        between `t_min` and `t_max`). If `initial_y` was a scalar, this is a 1D array of 
        shape `(n_steps,)`. If `initial_y` was an array-like structure of size N, this is a 2D matrix of 
        shape `(n_steps, N)`, where the k-th column is the solution to the k-th equation.
    x_values : ndarray
        A 1D array of shape `(n_steps,)` containing the corresponding integration steps 
        where the solution was evaluated. The final value is guaranteed to reach at least 
        `t_max`, but will not exceed it by more than `min_step` 
        (i.e., `x_values[-1] - t_max < min_step`).

    Raises
    ------
    TypeError
        If `function` is not a callable Python object or valid pointer,if `initial_y` 
        is not numeric, or if complex `extra_parameters` are provided alongside 
        `compile_to_C=True`.
    ValueError
        If `initial_y` is an empty array, a multi-dimensional matrix, or if an invalid 
        integer pointer is passed.
    """
    
    if not callable(function) and not isinstance(function, int) and not hasattr(function, 'address'):
        raise TypeError("The function must be a callable object or a C-pointer (int), or a Numba cfunc.")
        
    if extra_parameters is not None and not isinstance(extra_parameters, tuple):
        raise TypeError("Argument extra_parameters must be a tuple.")

    if np.isscalar(initial_y) and np.isreal(initial_y):
        initial_y = float(initial_y)    
    else:
        try:
            initial_y = np.ascontiguousarray(initial_y, dtype=np.float64)
        except Exception:
            raise TypeError("Argument initial_y must be a scalar float or a 1D array-like structure.")
            
        if initial_y.ndim != 1:
            raise ValueError(f"Argument initial_y must be a 1D vector.You passed a {initial_y.ndim}D array .")
        
        if initial_y.size == 0:
            raise ValueError("Argument initial_y cannot be an empty array.")


    simple_params = True
    if extra_parameters is not None:
        if not all(isinstance(p, (int, float)) for p in extra_parameters):
            simple_params = False

    ptrCompiled = None
    if compile_to_C or isinstance(function, int) or (hasattr(function, 'address') and isinstance(function.address, int)):
        
        if extra_parameters is not None and not simple_params:
            raise TypeError("You cannot use C-compiled function with complex extra_parameters like lists, strings,or dicts.Either set 'compile_to_C=False' or use global variables.")

        if isinstance(function, int):
            if function < 65000:
                raise ValueError(f"Invalid pointer: {function}. Pointer integers must be valid OS memory addresses.")
            ptrCompiled = function  
        elif hasattr(function, 'address') and isinstance(function.address, int):
            ptrCompiled = function.address
        else: 
            try:
                num_params = len(extra_parameters) if extra_parameters else 0
                if isinstance(initial_y, float):
                    compiled = _compile_scalar(function, num_params)
                    ptrCompiled = compiled.address
                    
                elif isinstance(initial_y, np.ndarray):
                    num_eq = len(initial_y)
                    compiled = _compile_vectorized(function, num_eq, num_params)
                    ptrCompiled = compiled.address

            except Exception as e:
                warnings.warn(f"Numba C-compilation failed, using user's python function directly.Compiler Error: {str(e)}",
                    category=RuntimeWarning,
                    stacklevel=2)

    target_func = ptrCompiled if ptrCompiled is not None else function
    
    return numeric.adaptiveRk(target_func, float(t_min), float(t_max), initial_y, float(tolerance), float(min_step), extra_parameters)