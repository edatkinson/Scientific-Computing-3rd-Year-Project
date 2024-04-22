import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from ode_solver import *
from Equations_Functions import *
from Diffusion_OO import DiffusionSimulation
from bvp_and_shooting import shoot, limit_cycle_finder, phase_condition
from sparse_dense_bvp import solve_equation, solve_bvp_root 
from numerical_continuation import *


import unittest

'''
Correctness of Steps: We verify that both the Euler and RK4 solvers compute the correct next step for a simple ODE dx/dt = -x. For the RK4 solver, I use the analytical solution for comparison because of its higher accuracy.
Zero Time Step: Tests to ensure that if the step size h is set to zero, the function returns the initial conditions unchanged, which is crucial for certain dynamic simulations.
Input Validation: Tests are included to check the robustness of the functions when provided with incorrect input types. This includes passing non-arrays for initial conditions, non-numeric types for times and step sizes, and non-callable functions.
'''

class TestODESolvers(unittest.TestCase):
    def setUp(self):
        # A simple ODE: dx/dt = -x
        self.test_func = lambda t, x: -x
        
        # Initial conditions and parameters
        self.x0 = np.array([1.0])  # Initial condition as a numpy array
        self.t0 = 0.0  # Initial time
        self.h = 0.1   # Step size

    def test_euler_step_correctness(self):
        """ Test the Euler step solver for correctness. """
        x1, t1 = euler_step(self.test_func, self.x0, self.t0, self.h)
        # The expected result for one Euler step: x1 = x0 + h * f(t0, x0) = 1 - 0.1 * 1
        np.testing.assert_allclose(x1, np.array([0.9]), atol=1e-5)

    def test_euler_step_zero_step(self):
        """ Test the Euler step solver with zero step size. """
        x1, t1 = euler_step(self.test_func, self.x0, self.t0, 0)
        np.testing.assert_array_equal(x1, self.x0)
        self.assertEqual(t1, self.t0)

    def test_euler_step_input_validation(self):
        """ Test the input validation in the Euler step solver. """
        with self.assertRaises(TypeError):
            euler_step(self.test_func, 'not_an_array', self.t0, self.h)
        with self.assertRaises(TypeError):
            euler_step(self.test_func, self.x0, 'not_a_number', self.h)
        with self.assertRaises(ValueError):
            euler_step('not_a_function', self.x0, self.t0, self.h)

    def test_rk4_step_correctness(self):
        """ Test the RK4 step solver for correctness. """
        x1, t1 = rk4_step(self.test_func, self.x0, self.t0, self.h)
        # The expected result for one RK4 step is more accurate than Euler's
        expected = self.x0 * np.exp(-self.h)  # Analytical solution at t0 + h
        np.testing.assert_allclose(x1, expected, atol=1e-5)

    def test_rk4_step_zero_step(self):
        """ Test the RK4 step solver with zero step size. """
        x1, t1 = rk4_step(self.test_func, self.x0, self.t0, 0)
        np.testing.assert_array_equal(x1, self.x0)
        self.assertEqual(t1, self.t0)

    def test_rk4_step_input_validation(self):
        """ Test the input validation in the RK4 step solver. """
        with self.assertRaises(TypeError):
            rk4_step(self.test_func, 'not_an_array', self.t0, self.h)
        with self.assertRaises(TypeError):
            rk4_step(self.test_func, self.x0, 'not_a_number', self.h)
        with self.assertRaises(ValueError):
            rk4_step('not_a_function', self.x0, self.t0, self.h)

class TestSolveODE(unittest.TestCase):
    def setUp(self):
        # Simple linear ODE: dx/dt = -x with known exponential decay solution
        self.simple_ode = lambda t, x: -x
        self.initial_x = np.array([1.0])  # Initial condition
        self.time_array = np.linspace(0, 1, 11)  # 10 intervals, 11 points from t=0 to t=1
        self.max_step = 0.1  # Maximum time step

    def test_solve_ode_euler_method(self):
        """ Test solve_ode using the Euler method over multiple time steps. """
        results = solve_ode(self.simple_ode, self.initial_x, self.time_array, "euler", self.max_step)
        expected = np.exp(-self.time_array)  # Analytical solution of the given ODE
        np.testing.assert_allclose(results.flatten(), expected, atol=1e-1)

    def test_solve_ode_rk4_method(self):
        """ Test solve_ode using the RK4 method over multiple time steps. """
        results = solve_ode(self.simple_ode, self.initial_x, self.time_array, "rk4", self.max_step)
        expected = np.exp(-self.time_array)  # Analytical solution of the given ODE
        np.testing.assert_allclose(results.flatten(), expected, atol=1e-3)

    def test_solve_ode_method_input_validation(self):
        """ Test solve_ode input validation for incorrect method input. """
        with self.assertRaises(ValueError):
            solve_ode(self.simple_ode, self.initial_x, self.time_array, "invalid_method", self.max_step)

    def test_solve_ode_time_array_validation(self):
        """ Test solve_ode handling of invalid time array inputs. """
        with self.assertRaises(TypeError):
            solve_ode(self.simple_ode, self.initial_x, "not_an_array", "euler", self.max_step)
        with self.assertRaises(TypeError):
            solve_ode(self.simple_ode, self.initial_x, np.array([[0, 1], [1, 2]]), "euler", self.max_step)

    def test_solve_ode_zero_max_step(self):
        """ Test solve_ode handling of zero maximum step size. """
        results = solve_ode(self.simple_ode, self.initial_x, self.time_array, "euler", 0)
        # Expect all results to be equal to the initial value since no steps should occur
        np.testing.assert_array_equal(results.flatten(), np.full_like(self.time_array, self.initial_x))



class TestODEAnalysisFunctions(unittest.TestCase):
    def setUp(self):
        # Define a simple test ODE: du/dt = -u, which has a well-known exponential decay solution
        self.simple_ode = lambda t, u, pars: -u
        # Initial condition
        self.u0 = np.array([1.0])
        # Parameters (not used in the simple ODE but required for the function signature)
        self.pars = ()

    def test_phase_condition(self):
        """Test the phase condition function which should return -1 derivative at t=0"""
        expected_phase_condition = np.array([-1])  # Because du/dt at t=0 is -u0 = -1
        result = phase_condition(self.simple_ode, self.u0, self.pars)
        np.testing.assert_array_equal(result, expected_phase_condition)

    def test_shoot(self):
        """Test the shoot function for correctly solving a BVP using the simple ODE"""
        def simple_phase_cond(ode, u0, pars):
            return np.array([ode(0, u0, pars)])

        # Define the shooting function with simple ODE and phase condition
        G = shoot(self.simple_ode, simple_phase_cond)
        
        # Use an initial guess close to the expected steady state
        initial_guess = np.array([0.1])
        T = 1  # Period of the limit cycle (dummy value for the test)
        
        # The expected result should be close to zero since the system should decay to zero
        result = G(initial_guess, T, self.pars)
        np.testing.assert_array_almost_equal(result[:-1], np.zeros_like(initial_guess), decimal=1)

    def test_limit_cycle_finder(self):
        """Test the limit_cycle_finder function for finding a trivial steady state using fsolve"""
        def trivial_phase_condition(ode, u0, pars):
            # A trivial phase condition that is always zero (for testing)
            return np.array([0])

        # Initial estimate includes a dummy period (1.0)
        initial_estimate = np.array([0.1, 1.0])
        solution = limit_cycle_finder(self.simple_ode, initial_estimate, trivial_phase_condition, self.pars)
        
        # Expected solution is that the steady state (u) approaches zero
        expected_solution = np.array([0.0, 1.0])  # Assume period remains unchanged
        np.testing.assert_array_almost_equal(solution[:-1], expected_solution[:-1], decimal=2)


class TestNumericalContinuation(unittest.TestCase):
    def setUp(self):
        # Define a simple test ODE: du/dt = a - bx, which has a known analytical behavior for small parameter ranges
        self.simple_ode = lambda t, x, params: x**3 - x -params[0]
        # Initial condition
        self.x0 = np.array([1.0])
        # Parameters (a=1, b=1 initially)
        self.par_array = [3, 1]
        # Index of the parameter 'a' to vary
        self.par_index = 0
        # Bounds for the parameter 'a' to be continued
        self.par_bounds = [2.8, 1.2]
        # Maximum steps for continuation
        self.max_steps = [70, 30]
        # Direction of continuation
        self.increase = False

    def test_numerical_continuation_natural(self):
        """Test the numerical_continuation function using the natural method."""
        pars, sol = numerical_continuation(
            self.simple_ode, 'natural', self.x0, self.par_array, self.par_index,
            self.par_bounds, self.max_steps, shoot, fsolve,
            None, self.increase
        )
        # Check if parameter values and solutions are correctly computed
        self.assertIsNotNone(pars)
        self.assertIsNotNone(sol)
        self.assertEqual(len(pars), 30)  # Expected number of steps
        self.assertEqual(len(sol), 30)

    def test_numerical_continuation_pseudo(self):
        """Test the numerical_continuation function using the pseudo method."""
        pars, sol = numerical_continuation(
            self.simple_ode, 'pseudo', self.x0, self.par_array, self.par_index,
            self.par_bounds, self.max_steps, shoot, fsolve,
            None, self.increase
        )
        # Check if parameter values and solutions are correctly computed
        self.assertIsNotNone(pars)
        self.assertIsNotNone(sol)
        self.assertEqual(len(pars), 70)  # Expected number of steps
        self.assertEqual(len(sol), 70)
    
    def test_numerical_continuation_pseudo_with_pc(self):
        """Test the numerical_continuation function using the pseudo method with a phase condition."""
        
        pars, sol = numerical_continuation(
            brusselator, 'pseudo', np.array([0.37, 3.5, 7.2]), [3,1], 0,
            [3,2.5], [73,30], shoot, fsolve,
            phase_condition=phase_condition, increase=self.increase
        )
        # Check if parameter values and solutions are correctly computed
        self.assertIsNotNone(pars)
        self.assertIsNotNone(sol)

    def test_numerical_continuation_invalid_method(self):
        """Test the numerical_continuation function with an invalid method."""
        with self.assertRaises(ValueError):
            numerical_continuation(
                self.simple_ode, 'invalid_method', self.x0, self.par_array, self.par_index,
                self.par_bounds, self.max_steps, None, fsolve,
                None, self.increase
            )


unittest.main()
