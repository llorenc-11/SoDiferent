import pytest
import math
import random
import numpy as np
import pypakiet.Numeric as rk

NUM_RANDOM_RUNS = 10

# --- Non-vectorized tests ---

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
    res_y, res_t = rk.adaptiveRk(decay_deriv, x_min=0, x_max=0, initial_y=5)
    assert len(res_t) == 1

def test_rk_adaptive_low_tolerance():
    """If the tolerance is very low, the minimal step value should take over and the loop shouldn't go on forever"""
    res_y,res_t = rk.adaptiveRk(decay_deriv,x_min=0,x_max=50,initial_y=0,tolerance=1e-15) 

    

def test_rk_adaptive_constant_derivative():
    """If y' = 0, y should remain exactly at the initial condition"""
    def zero_deriv(t, y):
        return 0

    res_y, res_t = rk.adaptiveRk(zero_deriv, x_min=0, x_max=100, initial_y=42, tolerance=1e-2)
    

    assert all(y == 42 for y in res_y)
    assert len(res_t) < 10


@pytest.mark.parametrize("execution_number", range(NUM_RANDOM_RUNS))
def test_rk_adaptive_decay_random(execution_number):
    """
    Test exponential function 
    Analytical solution: y(t) = y0 * e^{-(t - t0)}
    """
    t0 = random.uniform(0, 14)
    t_end = random.uniform(15, 115)
    y0 = random.uniform(0, 5)

    res_y, res_t = rk.adaptiveRk(decay_deriv, x_min=t0, x_max=t_end, initial_y=y0, tolerance=1e-8)

    assert len(res_y) > 0
    assert res_t[-1] == pytest.approx(t_end, abs=1e-3)
    
    exact_solve = y0 * math.exp(-(res_t[-2] - t0))
    assert res_y[-2] == pytest.approx(exact_solve, abs=1e-6)


@pytest.mark.parametrize("execution_number", range(NUM_RANDOM_RUNS))
def test_rk_adaptive_polynomial_random(execution_number):
    """
    Test for derivative dependent on only t
    Analytical solution: y(t) = t^2 - t0^2 + y0
    """

    t0 = random.uniform(0, 10)
    t_end = random.uniform(15, 30)
    y0 = random.uniform(-10, 10)

    res_y, res_t = rk.adaptiveRk(poly_deriv, x_min=t0, x_max=t_end, initial_y=y0, tolerance=1e-8)
    assert len(res_y) > 0
    assert res_t[-1] == pytest.approx(t_end, abs=1e-3)

    exact_solve = (res_t[-2]**2) - (t0**2) + y0
    assert res_y[-2] == pytest.approx(exact_solve, abs=1e-6)


@pytest.mark.parametrize("execution_number", range(NUM_RANDOM_RUNS))
def test_rk_adaptive_oscillatory_random(execution_number):
    """
    Testing periodic function,
    Analytical solution: y(t) = sin(t) - sin(t0) + y0
    """

    t0 = random.uniform(0, math.pi)
    t_end = random.uniform(math.pi + 1, 5 * math.pi)
    y0 = random.uniform(-3, 3)

    res_y, res_t = rk.adaptiveRk(trig_deriv, x_min=t0, x_max=t_end, initial_y=y0, tolerance=1e-8)
    
    assert len(res_y) > 0
    assert res_t[-1] == pytest.approx(t_end, abs=5e-3)

    exact_solve = math.sin(res_t[-2]) - math.sin(t0) + y0
    assert res_y[-2] == pytest.approx(exact_solve, abs=1e-6)




# --- Vectorized derivatives

def sys_zero_deriv(t, y):
    """y' = [0, 0, 0]"""
    return np.zeros_like(y)

def sys_decay_deriv(t, y):
    """
    Uncoupled system:
    y_1' = -y_1
    y_2' = -2 * y_2
    """
    return np.array([-y[0], -2.0 * y[1]])

def sys_harmonic_oscillator(t, y):
    """
    Coupled system (Spring/Pendulum):
    y_1' = y_2
    y_2' = -y_1
    """
    return np.array([y[1], -y[0]])


# --- Vectorized Tests ---

def test_rk_vector_zero_span():
    """Should return immediately with a 2D array if start and end times are identical"""
    y0 = np.array([5.0, 3.0])
    res_y, res_t = rk.adaptiveRk(sys_decay_deriv, x_min=0, x_max=0, initial_y=y0)
    
    assert len(res_t) == 1
    assert res_y.shape == (1, 2) 


def test_rk_vector_constant_derivative():
    """If y' = 0 vector, y should remain exactly at the initial condition for all equations"""
    y0 = np.array([42.0, -7.0, 3.14])
    res_y, res_t = rk.adaptiveRk(sys_zero_deriv, x_min=0, x_max=100, initial_y=y0, tolerance=1e-2)
    
    assert res_y.shape[1] == 3
    assert len(res_t) < 10

    expected_y = np.broadcast_to(y0, res_y.shape)
    
    np.testing.assert_allclose(res_y, expected_y)


@pytest.mark.parametrize("execution_number", range(NUM_RANDOM_RUNS))
def test_rk_vector_decay_random(execution_number):
    """
    Test uncoupled exponential decay
    Analytical solutions: 
    y1(t) = y1_0 * e^{-(t - t0)}
    y2(t) = y2_0 * e^{-2(t - t0)}
    """
    t0 = random.uniform(0, 14)
    t_end = random.uniform(15, 115)
    y0 = np.array([random.uniform(1, 5), random.uniform(1, 5)])

    res_y, res_t = rk.adaptiveRk(sys_decay_deriv, x_min=t0, x_max=t_end, initial_y=y0, tolerance=1e-8)

    assert len(res_y) > 0
    assert res_t[-1] == pytest.approx(t_end, abs=1e-3)
    
    t_eval = res_t[-2]
    dt = t_eval - t0
    
    exact_y1 = y0[0] * math.exp(-dt)
    exact_y2 = y0[1] * math.exp(-2.0 * dt)

    assert res_y[-2, 0] == pytest.approx(exact_y1, abs=1e-6)
    assert res_y[-2, 1] == pytest.approx(exact_y2, abs=1e-6)


@pytest.mark.parametrize("execution_number", range(NUM_RANDOM_RUNS))
def test_rk_vector_harmonic_oscillator_random(execution_number):
    """
    Harmonic Oscillator
    
    Analytical solution:
    y1(t) = y1_0 * cos(t - t0) + y2_0 * sin(t - t0)
    y2(t) = -y1_0 * sin(t - t0) + y2_0 * cos(t - t0)
    """
    t0 = random.uniform(0, math.pi)
    t_end = random.uniform(math.pi + 1, 5 * math.pi)
    y0 = np.array([random.uniform(-3, 3), random.uniform(-3, 3)])

    res_y, res_t = rk.adaptiveRk(sys_harmonic_oscillator, x_min=t0, x_max=t_end, initial_y=y0, tolerance=1e-8)
    
    assert len(res_y) > 0
    assert res_t[-1] == pytest.approx(t_end, abs=5e-3)

    t_eval = res_t[-2]
    dt = t_eval - t0
    
    exact_y1 = y0[0] * math.cos(dt) + y0[1] * math.sin(dt)
    exact_y2 = -y0[0] * math.sin(dt) + y0[1] * math.cos(dt)

    assert res_y[-2, 0] == pytest.approx(exact_y1, abs=1e-6)
    assert res_y[-2, 1] == pytest.approx(exact_y2, abs=1e-6)