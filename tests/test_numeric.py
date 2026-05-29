import pytest
import math
import random
import pypakiet.Numeric as rk

NUM_RANDOM_RUNS = 20

def decay_deriv(t, y):
    """y' = -y"""
    return -y

def poly_deriv(t, y):
    """y' = 2t"""
    return 2 * t

def trig_deriv(t, y):
    """y' = cos(t)"""
    return math.cos(t)


def test_rk_adaptive_basic():
    res_y,res_t = rk.adaptiveRk(decay_deriv,x_min=0,x_max=0,initial_y=0,tolerance=1e-5)
    assert len(res_t) == len(res_y)
    assert len(res_t) == 0
    res_y,res_t = rk.adaptiveRk(decay_deriv,x_min=0,x_max=50,initial_y=0,tolerance=1e-15) #infinite loop ? 
    




@pytest.mark.parametrize("execution_number", range(NUM_RANDOM_RUNS))
def test_rk_adaptive_decay_random(execution_number):
    """
    Test exponential decay. 
    Analytical solution: y(t) = y0 * e^{-(t - t0)}
    """
    t0 = random.uniform(0, 14)
    t_end = random.uniform(15, 115)
    y0 = random.uniform(0, 5)

    res_y, res_t = rk.adaptiveRk(decay_deriv, x_min=t0, x_max=t_end, initial_y=y0, tolerance=1e-5)

    assert len(res_t) == len(res_y)
    assert len(res_y) > 0
    assert res_t[-1] == pytest.approx(t_end, abs=1e-3)
    
    exact_solve = y0 * math.exp(-(t_end - t0))
    assert res_y[-1] == pytest.approx(exact_solve, abs=1e-3)


@pytest.mark.parametrize("execution_number", range(NUM_RANDOM_RUNS))
def test_rk_adaptive_polynomial_random(execution_number):
    """
    Test integration of a polynomial.
    Analytical solution: y(t) = t^2 - t0^2 + y0
    """

    t0 = random.uniform(0, 10)
    t_end = random.uniform(15, 50)
    y0 = random.uniform(-10, 10)

    res_y, res_t = rk.adaptiveRk(poly_deriv, x_min=t0, x_max=t_end, initial_y=y0, tolerance=1e-5)
    
    assert len(res_t) == len(res_y)
    assert len(res_y) > 0
    assert res_t[-1] == pytest.approx(t_end, abs=1e-3)

    exact_solve = (t_end**2) - (t0**2) + y0
    assert res_y[-1] == pytest.approx(exact_solve, abs=1e-3)


@pytest.mark.parametrize("execution_number", range(NUM_RANDOM_RUNS))
def test_rk_adaptive_oscillatory_random(execution_number):
    """
    Test integration of a periodic function.
    Analytical solution: y(t) = sin(t) - sin(t0) + y0
    """

    t0 = random.uniform(0, math.pi)
    t_end = random.uniform(math.pi + 1, 10 * math.pi)
    y0 = random.uniform(-5, 5)

    res_y, res_t = rk.adaptiveRk(trig_deriv, x_min=t0, x_max=t_end, initial_y=y0, tolerance=1e-5)
    
    assert len(res_t) == len(res_y)
    assert len(res_y) > 0
    assert res_t[-1] == pytest.approx(t_end, abs=1e-3)

    exact_solve = math.sin(t_end) - math.sin(t0) + y0
    assert res_y[-1] == pytest.approx(exact_solve, abs=1e-3)