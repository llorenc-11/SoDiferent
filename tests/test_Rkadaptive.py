import pytest
import math
import random
import warnings
import numpy as np
import pandas as pd 

import SoDiferent as sd

NUM_RANDOM_RUNS = 20


# --- Non-Vectorized Tests

def decay_deriv(t, y):
    """y' = -y"""
    return -y

def poly_deriv(t, y):
    """y' = 2t"""
    return 2 * t

def trig_deriv(t, y):
    """y' = cos(t)"""
    return math.cos(t)


def test_rk_adaptive_zero_span():
    """Should return immediately if start and end times are identical"""
    res_y, res_t = sd.RungeKutta(decay_deriv, t_min=0.0, t_max=0.0, initial_y=5.0)
    assert len(res_t) == 1

def test_rk_adaptive_low_tolerance():
    """If the tolerance is very low, the minimal step value should take over and the loop shouldn't go on forever"""
    res_y, res_t = sd.RungeKutta(decay_deriv, t_min=0.0, t_max=50.0, initial_y=0.0, tolerance=1e-15) 
    assert len(res_t) > 0

def test_rk_adaptive_constant_derivative():
    """If y' = 0, y should remain exactly at the initial condition"""
    def zero_deriv(t, y):
        return 0.0

    res_y, res_t = sd.RungeKutta(zero_deriv, t_min=0.0, t_max=100.0, initial_y=42.0, tolerance=1e-2)
    assert all(y == 42.0 for y in res_y)
    assert len(res_t) < 10


@pytest.mark.parametrize("execution_number", range(NUM_RANDOM_RUNS))
def test_rk_adaptive_decay_random(execution_number):
    """Test exponential function. Analytical solution: y(t) = y0 * e^{-(t - t0)}"""
    t0 = random.uniform(0, 14)
    t_end = random.uniform(15, 115)
    y0 = random.uniform(0, 5)

    res_y, res_t = sd.RungeKutta(decay_deriv, t_min=t0, t_max=t_end, initial_y=y0, tolerance=1e-8)

    assert len(res_y) > 0
    assert res_t[-1] == pytest.approx(t_end, abs=1e-3)
    
    exact_solve = y0 * math.exp(-(res_t[-2] - t0))
    assert res_y[-2] == pytest.approx(exact_solve, abs=1e-6)

@pytest.mark.parametrize("execution_number", range(NUM_RANDOM_RUNS))
def test_rk_adaptive_polynomial_random(execution_number):
    """Test for derivative dependent on only t. Analytical solution: y(t) = t^2 - t0^2 + y0"""
    t0 = random.uniform(0, 10)
    t_end = random.uniform(15, 30)
    y0 = random.uniform(-10, 10)

    res_y, res_t = sd.RungeKutta(poly_deriv, t_min=t0, t_max=t_end, initial_y=y0, tolerance=1e-8)
    assert len(res_y) > 0
    assert res_t[-1] == pytest.approx(t_end, abs=1e-3)

    exact_solve = (res_t[-2]**2) - (t0**2) + y0
    assert res_y[-2] == pytest.approx(exact_solve, abs=1e-6)

# --- Vectorised Tests

def sys_zero_deriv(t, y):
    """y' = [0, 0, 0]"""
    return np.zeros_like(y)

def sys_zero_deriv_v2(t, y):
    """y' = [0, 0, 0]"""
    return [0 for i in range(len(y))]

def sys_decay_deriv(t, y):
    """Uncoupled system"""
    return np.array([-y[0], -2.0 * y[1]])

def sys_harmonic_oscillator(t, y):
    """Coupled system (Spring/Pendulum)"""
    return np.array([y[1], -y[0]])

def test_rk_vector_zero_witdh():
    """Should return immediately with a 2D array if start and end times are identical"""
    y0 = np.array([5.0, 3.0])
    res_y, res_t = sd.RungeKutta(sys_decay_deriv, t_min=0.0, t_max=0.0, initial_y=y0)
    
    assert len(res_t) == 1
    assert res_y.shape == (1, 2) 

def test_rk_vector_constant_derivative():
    """If y' = 0 vector, y should remain exactly at the initial condition for all equations"""
    y0 = np.array([42.0, -7.0, 3.14])
    res_y, res_t = sd.RungeKutta(sys_zero_deriv, t_min=0.0, t_max=100.0, initial_y=y0, tolerance=1e-2)
    
    assert res_y.shape[1] == 3

    expected_y = np.broadcast_to(y0, res_y.shape)
    np.testing.assert_allclose(res_y, expected_y)

@pytest.mark.parametrize("initial_y_input",
    [[1, 3, 7], 
    (1, 3, 7), 
    pd.Series([1, 3, 7]),
    np.array([1, 3, 7])])
def test_array_like_structures_safely_parsed(initial_y_input):
    """
    Check if the Python wrapper correctly catches all array-like structures and changes them corectly before they cross to C++ solver
    """
    # 1. Run the solver (using the slow path since compile_to_C=False by default)
    res_y, res_t = sd.RungeKutta(
        function=sys_zero_deriv_v2, 
        t_min=0.0, 
        t_max=12.0, 
        initial_y=initial_y_input,
        compile_to_C = True
    )

    assert len(res_t) > 0
    assert res_y.shape[1] == 3


    expected_y = np.broadcast_to([1.0, 3.0, 7.0], res_y.shape)
    np.testing.assert_allclose(res_y, expected_y, atol=1e-10)




@pytest.mark.parametrize("execution_number", range(NUM_RANDOM_RUNS))
def test_rk_vector_harmonic_oscillator_random(execution_number):
    """
    Harmonic Oscillator
    
    Analytical solution:
    y1(t)= y1_0* cos(t - t0) + y2_0 *sin(t - t0)
    y2(t) = -y1_0 *sin(t - t0) +y2_0 * cos(t - t0)
    """
    t0 = random.uniform(0, math.pi)
    t_end = random.uniform(math.pi + 1, 5 * math.pi)
    y0 = np.array([random.uniform(-3, 3), random.uniform(-3, 3)])

    res_y, res_t = sd.RungeKutta(sys_harmonic_oscillator, t_min=t0, t_max=t_end, initial_y=y0, tolerance=1e-8)
    
    assert len(res_y) > 0
    assert res_t[-1] == pytest.approx(t_end, abs=5e-3)

    t_eval = res_t[-2]
    dt = t_eval - t0
    
    exact_y1 = y0[0] * math.cos(dt) + y0[1] * math.sin(dt)
    exact_y2 = -y0[0] * math.sin(dt) + y0[1] * math.cos(dt)

    assert res_y[-2, 0] == pytest.approx(exact_y1, abs=1e-6)
    assert res_y[-2, 1] == pytest.approx(exact_y2, abs=1e-6)

# --- Numba Tests

def test_fast_path_simple_parameters():
    def parameterized_decay(t, y, params):
        rate = params[0] 
        return -rate * y

    y0 = 5.0
    custom_rate = 2.5
    
    res_y, res_t = sd.RungeKutta(
        parameterized_decay, 
        t_min=0.0, 
        t_max=4.0, 
        initial_y=y0, 
        tolerance=1e-8, 
        extra_parameters=(custom_rate,),
        compile_to_C=True
    )
    
    assert len(res_y) > 0
    exact_solve = y0 * math.exp(-custom_rate * res_t[-1])
    assert res_y[-1] == pytest.approx(exact_solve, abs=1e-6)

def test_slow_path_complex_parameters():

    def dirty_decay(t, y, params):
        config_dict = params[0] 
        return -config_dict["rate"] * y

    y0 = 5.0
    res_y, res_t = sd.RungeKutta(
        dirty_decay, 
        t_min=0.0, 
        t_max=5.0, 
        initial_y=y0, 
        extra_parameters=({"rate": 1.0},),
        compile_to_C=False
    )
    assert len(res_y) > 0


# --- Others Tests

def test_compile_mismatch_complex_params():
    """ Ensures the code catches users trying to pass lists/strings while simultaneously asking for C compilation """
    with pytest.raises(TypeError, match="You cannot use C-compiled function with complex extra_parameters"):
        sd.RungeKutta(
            decay_deriv, 
            t_min=0.0, 
            t_max=1.0, 
            initial_y=1.0, 
            extra_parameters=(["list", "of", "strings"],), 
            compile_to_C=True
        )

def test_invalid_pointer_integer():
    """ Ensures that random, low-value integers are blocked from crossing into C++ to prevent a Segmentation Fault """
    with pytest.raises(ValueError, match="Invalid pointer"):
        sd.RungeKutta(
            function=3, #Garbage 
            t_min=0.0, 
            t_max=1.0, 
            initial_y=5.0,
            compile_to_C=True
        )

def test_guard_numba_fallback_warning():
        """ Ensures that if Numba compilation fails a warning is shown and safely falls back to Python """
        def unsupported_math(t, y):
            return eval("1.0") * y

        with pytest.warns(RuntimeWarning, match="Numba C-compilation failed"):
            res_y, res_t = sd.RungeKutta(
                unsupported_math, 
                t_min=0.0, 
                t_max=2.0, 
                initial_y=5.0, 
                compile_to_C=True
            )
        assert len(res_y) > 0

def test_invalid_initial_y_matrix():
    """ Checks if initial_y is a 1D vector otherwise throws an error """
    bad_y0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="must be a 1D vector"):
        sd.RungeKutta(sys_decay_deriv, t_min=0.0, t_max=1.0, initial_y=bad_y0)


