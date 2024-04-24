import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from ode_solver import *
from Equations_Functions import *
from Diffusion_OO import *
from bvp_and_shooting import shoot, limit_cycle_finder, phase_condition
from sparse_dense_bvp import *
from numerical_continuation import *
from scipy.sparse import diags, csr_matrix

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

import unittest


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
        T = 1  # Period of the limit cycle 
        
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
        self.par_array = [3, 1]
        self.par_index = 0
        self.par_bounds = [2.8, 1.2]
        self.max_steps = [70, 30]
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


class TestFiniteDifferenceSetup(unittest.TestCase):
    def setUp(self):
        self.n_points = 10
        self.h = 0.1
        self.bc_left = BoundaryCondition('left', 'dirichlet', 0)
        self.bc_right = BoundaryCondition('right', 'dirichlet', 0)
        self.coefficients = {'D': 1.0}
    
    def test_diffusion_against_analytical(self):
        L = 1  # Length of the domain
        N = 100  # Number of points
        D = 1.0  # Diffusion coefficient
        x = np.linspace(0, L, N)
        h = x[1] - x[0]

        # Analytical solution
        u_analytical = -1/(2*D) * (x)*(x-1) # Analytical solution for the diffusion equation (from week 19) with rhs = 1

        u_sparse = solve_equation('sparse', lambda n_points, coefficients, domain: -np.ones(len(domain)), x, h, self.bc_left, self.bc_right, self.coefficients, 'diffusion')

        # Comparing the numerical solution with the analytical solution
        np.testing.assert_allclose(u_sparse, u_analytical, atol=1e-4, err_msg="Numerical solution does not match analytical solution closely enough.")

    def test_matrix_setup_diffusion(self):
        A = setup_finite_difference_matrix(self.n_points, self.h, 'diffusion', self.bc_left, self.bc_right, self.coefficients, 'sparse')
        # Check the main diagonal and off-diagonal values
        self.assertAlmostEqual(A[0, 0], -2 * self.coefficients['D'] / self.h**2)
        self.assertAlmostEqual(A[0, 1], self.coefficients['D'] / self.h**2)
        self.assertAlmostEqual(A[1, 0], self.coefficients['D'] / self.h**2)

    def test_boundary_conditions_application(self):
        A = csr_matrix((self.n_points, self.n_points))
        A = apply_boundary_conditions(A, self.bc_left, self.bc_right, self.h)
        # Check boundary application
        self.assertEqual(A[0, 0], 1.0)
        self.assertEqual(A[-1, -1], 1.0)

    def test_solution_accuracy(self):
        domain = np.linspace(-1, 1, self.n_points)
        rhs_function = lambda n_points, coefficients, domain: np.ones(n_points)  # Simple RHS for testing
        u_dense = solve_dense(rhs_function, domain, self.h, self.bc_left, self.bc_right, self.coefficients, 'diffusion')
        u_sparse = solve_sparse(rhs_function, domain, self.h, self.bc_left, self.bc_right, self.coefficients, 'diffusion')

        # Assert that both solutions are nearly identical
        np.testing.assert_allclose(u_dense, u_sparse, atol=1e-5)

    def test_matrix_setup_convection(self):
        self.coefficients = {'P': 0.5}  # Convection coefficient
        A = setup_finite_difference_matrix(self.n_points, self.h, 'convection', self.bc_left, self.bc_right, self.coefficients, 'sparse')
        # Check the main diagonal and off-diagonal adjustments for convection
        self.assertAlmostEqual(A[1, 1], -self.coefficients['P'] / self.h)
        self.assertAlmostEqual(A[1, 0], self.coefficients['P'] / self.h)
    
    def test_mixed_boundary_conditions_1(self):
        bc_left = BoundaryCondition('left', 'neumann', 0.5)
        bc_right = BoundaryCondition('right', 'dirichlet', -1.0)
        A = setup_finite_difference_matrix(self.n_points, self.h, 'diffusion', bc_left, bc_right, self.coefficients, 'sparse')
        A = apply_boundary_conditions(A, bc_left, bc_right, self.h)
        # Check Neumann at left boundary
        self.assertAlmostEqual(A[0, 0], -1 / self.h)
        self.assertAlmostEqual(A[0, 1], 1 / self.h)
        # Check Dirichlet at right boundary
        self.assertEqual(A[-1, -1], 1.0)
    
    def test_mixed_boundary_conditions_2(self):
        bc_left = BoundaryCondition('left', 'dirichlet', 0.5)
        bc_right = BoundaryCondition('right', 'neumann', -1.0)
        A = setup_finite_difference_matrix(self.n_points, self.h, 'diffusion', bc_left, bc_right, self.coefficients, 'sparse')
        A = apply_boundary_conditions(A, bc_left, bc_right, self.h)
        # Check Neumann at right boundary
        self.assertAlmostEqual(A[-1, -1], 1 / self.h)
        self.assertAlmostEqual(A[-1, -2], -1 / self.h)
        # Check Dirichlet at left boundary
        self.assertEqual(A[0, 0], 1.0)
    
    def test_parameter_sensitivity(self):
        high_diffusion = {'D': 10.0}
        A_high = setup_finite_difference_matrix(self.n_points, self.h, 'diffusion', self.bc_left, self.bc_right, high_diffusion, 'sparse')
        self.assertAlmostEqual(A_high[0, 0], -2 * high_diffusion['D'] / self.h**2)
        self.assertAlmostEqual(A_high[0, 1], high_diffusion['D'] / self.h**2)

    def test_small_and_large_grid_sizes(self):
        for n in [5, 1000]:  # Very small and very large grids
            A = setup_finite_difference_matrix(n, self.h, 'diffusion', self.bc_left, self.bc_right, self.coefficients, 'sparse')
            self.assertEqual(A.shape, (n, n))  # Matrix should always be square of size n

    def test_invalid_equation_type(self):
        with self.assertRaises(ValueError):
            solve_equation('sparse', lambda n_points, coefficients, domain: np.ones(n_points), np.linspace(-1, 1, self.n_points), self.h, self.bc_left, self.bc_right, self.coefficients, 'unknown_type')

    def test_invalid_boundary_condition(self):
        bc_invalid = BoundaryCondition('left', 'unknown_type', 1.0)
        with self.assertRaises(ValueError):
            apply_boundary_conditions(csr_matrix((self.n_points, self.n_points)), bc_invalid, self.bc_right, self.h)
        with self.assertRaises(ValueError):
            apply_boundary_conditions(csr_matrix((self.n_points, self.n_points)), self.bc_left, bc_invalid, self.h)
    
    def test_invalid_matrix_type(self):
        with self.assertRaises(ValueError):
            solve_equation('unknown_type', lambda n_points, coefficients, domain: np.ones(n_points), np.linspace(-1, 1, self.n_points), self.h, self.bc_left, self.bc_right, self.coefficients, 'diffusion')
    
    def test_invalid_rhs_function(self):
        with self.assertRaises(TypeError):
            solve_equation('sparse',  'Not_rhs', self.h, self.bc_left, self.bc_right, self.coefficients, 'diffusion')

    def test_invalid_coefficients(self):
        with self.assertRaises(TypeError):
            solve_equation('sparse', lambda n_points, coefficients, domain: np.ones(n_points), np.linspace(-1, 1, self.n_points), self.h, self.bc_left, self.bc_right, {'D':'2'}, 'diffusion')


def analytical_solution(x,t,D,a,b):
    return np.exp(-D * np.pi ** 2 * t / ((b - a) ** 2)) * np.sin(np.pi * (x - a) / (b - a))

class TestDiffusionSimulation(unittest.TestCase):
    def setUp(self):
        """Setup a common simulation scenario for each test."""
        self.a = 0
        self.b = 1
        self.D = 1.0
        self.N = 50
        self.dx = (self.b - self.a) / (self.N - 1)
        self.initial_condition = lambda x: np.sin(np.pi * x)
        self.boundary_conditions = [
            BoundaryCondition('left', 'dirichlet', 0),
            BoundaryCondition('right', 'dirichlet', 0)
        ]
        self.time_span = (0, 1)
        self.source_term = lambda t, x, U: 0  # No source term for simplicity
        self.method = 'explicit_euler'
        dt_max = ((self.b-self.a)/self.N)**2/ (2 * self.D)
        self.dt = 0.5*dt_max
        self.simulation = DiffusionSimulation(
            self.source_term, self.a, self.b, self.D, self.initial_condition,
            self.boundary_conditions, self.N, self.time_span, self.method,self.dt
        )

    def test_diffusion_rhs(self):
        """Test the calculation of the diffusion RHS."""
        U = np.linspace(0, 1, self.N)
        t = 0
        rhs = self.simulation.diffusion_rhs(t, U)
        expected_rhs = np.zeros_like(U)  # Since U is linear and boundary conditions are zero
        np.testing.assert_allclose(rhs, expected_rhs, atol=1e-5)

    def test_apply_boundary_conditions(self):
        """Test boundary condition application."""
        U = np.array([1, 2, 3, 4, 5])
        self.simulation.apply_boundary_conditions(U)
        self.assertEqual(U[0], 0)
        self.assertEqual(U[-1], 0)

    def test_explicit_euler_step(self):
        """Test the explicit Euler step functionality."""
        U = self.initial_condition(np.linspace(self.a, self.b, self.N))
        t = 0
        U_next = self.simulation.explicit_euler_step(t, U)
        # Assuming a very small dt and simple diffusion, this should be close to initial U
        np.testing.assert_allclose(U_next, U, atol=1e-1)

    def test_steady_state_solution(self):
        """Test the steady state solver against an expected result (if applicable)."""
        x, U_steady = self.simulation.solve_steady_state()
        # Assuming a test scenario where analytical solution is known:
        analytical = np.zeros_like(x)  # for a trivial steady state case
        np.testing.assert_allclose(U_steady, analytical, atol=1e-5)

    def test_solver_methods(self):
        """Test different solver methods to ensure they compute correctly against analytical solution defined above."""
        methods = ['explicit_euler', 'implicit_dense', 'implicit_sparse','IMEX','implicit_root']
        for method in methods:
            sim = DiffusionSimulation(
                self.source_term, self.a, self.b, self.D, self.initial_condition,
                self.boundary_conditions, self.N, self.time_span, method,self.dt
            )
            x, t, U = sim.solve()
            error_threshold = 0.1  # Define a suitable error threshold
            for i, t in enumerate(np.arange(self.time_span[0], self.time_span[1], self.dt)):
                u_exact = analytical_solution(np.linspace(self.a,self.b,self.N+1), t,self.D,self.a,self.b)
                # Compute the error norm between the numerical and analytical solutions
                error_norm = np.linalg.norm(U[i, :] - u_exact, ord=2)
                self.assertLess(error_norm, error_threshold, f"Exceeded error threshold at t={t}")
                
            print(f'Method {method} passed test')


unittest.main()
