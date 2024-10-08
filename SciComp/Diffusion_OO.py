import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve
import matplotlib.pyplot as plt
from scipy.optimize import root, fsolve
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
import time
import cProfile
import warnings


class DiffusionSimulation:
    """
        A class to simulate the diffusion process governed by a partial differential equation.

        Attributes:
            a (float): The starting point of the spatial domain.
            b (float): The end point of the spatial domain.
            D (float): The diffusion coefficient.
            initial_condition (callable): A function that provides the initial condition of the system.
            boundary_conditions (list): A list of BoundaryCondition instances defining the boundary conditions.
            N (int): The number of spatial points in the discretization.
            dx (float): The spatial step size, calculated from `a`, `b`, and `N`.
            time_span (tuple): A tuple containing the start and end times for the simulation.
            method (str): The numerical method to use for the time integration ('explicit_euler', 'implicit_dense', etc.).
            dt (float): The time step size.
            source_term (callable): A function representing the source term of the PDE.

        Methods:
            diffusion_rhs(t, U): Computes the right-hand side of the diffusion equation.
            apply_boundary_conditions(U, A=None, F=None): Applies the boundary conditions to the solution vector or matrix.
            explicit_euler_step(t, U): Performs a time step using the explicit Euler method.
            create_sparse_matrix(N, dt, D, dx): Creates a sparse matrix for the implicit Euler method.
            implicit_euler_step(t, U, storage='dense'): Performs a time step using the implicit Euler method with optional storage mode.
            diffusion_solver(method): Solves the diffusion equation using IMEX of Implicit_Root method over the entire time span.
            solve_u_next(U_prev, source_term_func, x, n): Solves for the next time step's solution vector.
            solve_steady_state(): Solves for the steady state solution of the diffusion equation.
            solve(): Solves the diffusion equation over the specified time span using the specified method.
    """
    def __init__(self, source_term, a, b, D, initial_condition, boundary_conditions, N, time_span, method, dt):
        """
        Initializes the DiffusionSimulation class with the provided parameters and asserts the validity of inputs.
        """
        assert callable(source_term), "Source term must be callable."
        assert isinstance(a, (int, float)), "a must be an integer or float."
        assert isinstance(b, (int, float)), "b must be an integer or float."
        assert isinstance(D, (int, float)), "D must be an integer or float."
        assert callable(initial_condition), "Initial condition must be callable."
        assert all(isinstance(bc, BoundaryCondition) for bc in boundary_conditions), "Each boundary condition must be an instance of BoundaryCondition."
        assert isinstance(N, int) and N > 1, "N must be an integer greater than 1."
        assert isinstance(time_span, tuple) and len(time_span) == 2, "Time span must be a tuple of two elements."
        assert isinstance(method, str), "Method must be a string."
        assert isinstance(dt, (int, float)), "dt must be an integer or float."

        self.a = a
        self.b = b
        self.D = D
        self.initial_condition = initial_condition
        self.boundary_conditions = boundary_conditions
        self.N = N
        self.dx = (b - a) / (N-1)
        self.time_span = time_span
        self.method = method
        self.dt = dt
        self.source_term = source_term

    def diffusion_rhs(self,t, U):
        """
        Computes the right-hand side of the diffusion equation using central differences.

        Parameters:
            t (float): The current time.
            U (numpy.ndarray): The current values of the solution at each spatial point.

        Returns:
            numpy.ndarray: The spatial derivatives of U.
        """
        # Initialize the derivative vector
        dUdt = np.zeros_like(U)
        x = np.linspace(self.a, self.b, len(U))
        # Apply central difference for interior points using vector methods instead of loops
        dUdt[1:-1] = self.D * (U[2:] - 2*U[1:-1] + U[:-2]) / self.dx**2 + self.source_term(t, x[1:-1], U[1:-1])
        return dUdt

    def apply_boundary_conditions(self, U, A=None, F=None):
        """
        Applies the boundary conditions to the solution vector or matrix.

        Parameters:
            U (numpy.ndarray): The current values of the solution at each spatial point.
            A (numpy.ndarray, optional): The coefficient matrix (used in implicit methods).
            F (numpy.ndarray, optional): The modified right-hand side vector after applying source terms.
        """
        for bc in self.boundary_conditions:
            bc.apply(U, A, F, self.dx, self.D, self.dt) #apply BCs

    def explicit_euler_step(self, t,U):
        """
        Performs a time step using the explicit Euler method.

        Parameters:
            t (float): The current time.
            U (numpy.ndarray): The current values of the solution at each spatial point.

        Returns:
            numpy.ndarray: The updated values of U after one time step.
        """
        dUdt = self.diffusion_rhs(t,U)
        U_new = U + self.dt * dUdt # Update U using the explicit Euler method
        self.apply_boundary_conditions(U_new)
        return U_new

    def create_sparse_matrix(self, N, dt, D, dx):
        """
        Constructs a sparse matrix representing the discretized form of a diffusion operator
        using the finite difference method, suitable for numerical PDE simulations.

        This method creates a sparse matrix that combines the identity matrix and a scaled Laplacian matrix,
        used primarily in solving time-dependent diffusion equations numerically.

        Args:
            N (int): Number of grid points, determining the size of the matrix.
            dt (float): Time step size used in the discretization of the time derivative.
            D (float): Diffusion coefficient, a parameter of the diffusion equation.
            dx (float): Spatial step size used in the discretization of the spatial derivative.

        Returns:
            scipy.sparse.csr_matrix: A sparse matrix in Compressed Sparse Row format. This matrix can be used
                                    in numerical simulations to represent the linear part of the discretized
                                    PDE, effectively combining the identity matrix with a diffusion term scaled
                                    appropriately by time and spatial step sizes.
        """
        main_diag = -2 * np.ones(N)
        off_diag = np.ones(N - 1)
        
        diagonals = [main_diag, off_diag, off_diag]
        offsets = [0, -1, 1]
        laplacian = diags(diagonals, offsets, format='csr')
        
        laplacian_scaled = ((-dt * D) / dx**2) * laplacian
        
        identity = eye(N, format='csr')
        
        A = identity + laplacian_scaled
        
        return A
 
    def implicit_euler_step(self, t,U, storage='dense'):
        """
        Performs a time step using the implicit Euler method with the choice of storage scheme for the matrix.

        Parameters:
            t (float): The current time.
            U (numpy.ndarray): The current values of the solution at each spatial point.
            storage (str): The type of storage for the matrix ('dense' or 'sparse').

        Returns:
            numpy.ndarray: The updated values of U after one time step.
        """
        N = len(U)
        F = U.copy() + self.dt *self.source_term(t, np.linspace(self.a, self.b, N), U) 
        if storage == 'dense':
            A = np.eye(N) - self.dt * self.D * (np.diag(np.ones(N-1), -1) - 2*np.diag(np.ones(N), 0) + np.diag(np.ones(N-1), 1)) / self.dx**2
            self.apply_boundary_conditions(U, A, F)
            U_new = solve(A, F) #solve using dense matrix
        else:
            A_sparse = self.create_sparse_matrix(N, self.dt, self.D, self.dx)
            self.apply_boundary_conditions(U, A_sparse, F)
            U_new = spsolve(A_sparse, F) #solve using sparse matrix

        if storage == 'sparse' and N < 500:
            warnings.warn("Using a sparse solver for a small system may not be optimal.", RuntimeWarning)


        return U_new
    
    def diffusion_solver(self, method):
        """
        Solves the diffusion equation using the specified method over the entire time span.

        Parameters:
            method (str): The numerical method to use ('IMEX', 'implicit_root').

        Returns:
            tuple: The spatial points (x) and the matrix of solutions (u) over time.
        """
        Nt = int(self.time_span[1] / self.dt) + 1
        Nu = self.N + 1
        u = np.zeros((Nt, Nu))
        x = np.linspace(self.a, self.b, Nu)
        u[0, :] = self.initial_condition(x)

        for n in range(1, Nt):
            U_prev = u[n-1, :]

            if method == 'IMEX':
                # Compute the source term explicitly
                source_term = self.source_term(n * self.dt, x, U_prev)
                source_term_func = lambda U_next: source_term  # Explicit source term does not depend on U_next
            elif method == 'implicit_root':
                # Compute the source term implicitly
                source_term_func = lambda U_next: self.source_term(n * self.dt, x, U_next)
            else:
                raise ValueError("Unsupported method. Choose 'IMEX' or 'implicit_root'.")

            U_next = self.solve_u_next(U_prev, source_term_func, x, n)
            u[n, :] = U_next

        return x, u

    def solve_u_next(self, U_prev, source_term_func, x, n):
        """
        Solves for the next time step's solution vector using a root-finding method.

        Parameters:
            U_prev (numpy.ndarray): The solution vector from the previous time step.
            source_term_func (callable): A function that computes the source term.
            x (numpy.ndarray): The array of spatial points.
            n (int): The current time step index.

        Returns:
            numpy.ndarray: The solution vector for the next time step.
        """
        def equation(U_next):
            # Calculate diffusion term using central difference
            diffusion_term = np.zeros_like(U_next)
            diffusion_term[1:-1] = self.D * (U_next[:-2] - 2 * U_next[1:-1] + U_next[2:]) / self.dx**2
            non_linear_term = source_term_func(U_next)
            F = U_next - U_prev - self.dt * (diffusion_term + non_linear_term)
            # Apply boundary conditions to the residual vector
            self.apply_boundary_conditions(U_next, F=F)
            return F

        # Use a root-finding method to get U_next
        return root(equation, U_prev).x
 

    def solve_steady_state(self):
        """
        Solves for the steady state solution of the diffusion equation.

        Returns:
            tuple: The spatial points (x) and the steady state solution.
        """
        def equations(U):
            d2Udx2 = np.zeros_like(U)
            x = np.linspace(self.a, self.b, len(U))
            d2Udx2[1:-1] = self.D * (U[2:] - 2 * U[1:-1] + U[:-2]) / self.dx**2 + self.source_term(0, x[1:-1], U[1:-1])
            
            # Apply boundary conditions
            for bc in self.boundary_conditions:
                if bc.position == 'left':
                    if bc.type == 'dirichlet':
                        d2Udx2[0] = U[0] - bc.value
                    elif bc.type == 'neumann':
                        d2Udx2[0] = (U[1] - U[0]) / self.dx - bc.value
                elif bc.position == 'right':
                    if bc.type == 'dirichlet':
                        d2Udx2[-1] = U[-1] - bc.value
                    elif bc.type == 'neumann':
                        d2Udx2[-1] = (U[-1] - U[-2]) / self.dx - bc.value
            
            return d2Udx2
    
        x = np.linspace(self.a, self.b, self.N+1)
        initial_guess = self.initial_condition(x)
        steady_state_sol = fsolve(equations, initial_guess)
        return x, steady_state_sol
    

    def solve(self):
        """
        Solves the diffusion equation over the specified time span using the specified method.

        Returns:
            tuple: The spatial points (x), the time points (t), and the matrix of solutions over time.
        """
        x = np.linspace(self.a, self.b, self.N+1)
        
        U = self.initial_condition(x)
        timesteps = int((self.time_span[1] - self.time_span[0]) / self.dt)
        t = np.linspace(self.time_span[0], self.time_span[1], timesteps+1)
        U_sol = [U.copy()]
        current_t = self.time_span[0]
        if self.method == 'implicit_root':
            x, u = self.diffusion_solver(self.method)
            return x, t, u
        elif self.method == 'IMEX':
            x, u = self.diffusion_solver(self.method)
            return x, t, u
        else:
            for i in range(timesteps): #time stepping
                if self.method == 'explicit_euler':
                    if self.dt > self.dx**2 / (2 * self.D):
                        raise ValueError(f"Explicit Euler method is unstable for dt > dx^2 / (2 * D). Current dt={self.dt} and dx={self.dx}.")
                    U = self.explicit_euler_step(current_t, U)
                elif self.method == 'implicit_dense':
                    U = self.implicit_euler_step(current_t, U, storage = 'dense')
                elif self.method == 'implicit_sparse':
                    U = self.implicit_euler_step(current_t, U, storage = 'sparse')
                else:
                    raise ValueError("Unsupported method. Choose 'explicit_euler', 'implicit_dense' or 'implicit_sparse', 'implicit_root' or 'IMEX'.")
                U_sol.append(U.copy())
                current_t += self.dt
            return x, np.linspace(self.time_span[0], self.time_span[1], timesteps+1), np.array(U_sol)

class BoundaryCondition:
    """
    A class to represent a boundary condition for the diffusion equation.

    Attributes:
        position (str): The position of the boundary ('left' or 'right').
        type (str): The type of boundary condition ('dirichlet' or 'neumann').
        value (float): The value of the boundary condition.
        coefficients (list): The coefficients for a Robin boundary condition (alpha, beta).
    
    Methods:
        apply(U, A=None, F=None, dx=None, D=None, dt=None): Applies the boundary condition to the solution vector or matrix.
        

    """
    def __init__(self, position, type, value, coefficients=None):
        """
        Initializes the BoundaryCondition class with the provided parameters and asserts the validity of inputs.
        """

        assert isinstance(position, str), "Position must be a string."
        assert isinstance(type, str), "Type must be a string."
        assert isinstance(value, (int, float)), "Value must be an integer or float."
        if coefficients is not None:
            assert all(isinstance(c, (int, float)) for c in coefficients), "Coefficients must be a list of numbers."

        self.position = position
        self.type = type
        self.value = value
        self.coefficients = coefficients #coefficients for Robin boundary condition (alpha, beta)

    def apply(self, U, A=None, F=None, dx=None, D=None, dt=None):
        """
        Applies the boundary condition to the solution vector or matrix.
        
        Parameters:
            U (numpy.ndarray): The current values of the solution at each spatial point.
            A (numpy.ndarray, optional): The coefficient matrix (used in implicit methods).
            F (numpy.ndarray, optional): The modified right-hand side vector after applying source terms.
            dx (float, optional): The spatial step size.
            D (float, optional): The diffusion coefficient.
            dt (float, optional): The time step size.
        """

        index = 0 if self.position == 'left' else -1
        if self.type == 'dirichlet':
            if A is not None:
                A[index, :] = 0
                A[index, index] = 1
                F[index] = self.value
            U[index] = self.value
        elif self.type == 'neumann':
            if A is not None:
                # Adjusting A and F for Neumann boundary condition.
                if self.position == 'left':
                    A[0, 0] = -1
                    A[0, 1] = 1
                    F[0] = self.value * dx
                elif self.position == 'right':
                    A[-1, -1] = 1
                    A[-1, -2] = -1
                    F[-1] = -self.value * dx
            else:
                # Assuming a ghost node approach for the explicit method.
                if self.position == 'left':
                    U[0] = U[1] - self.value * dx
                elif self.position == 'right':
                    U[-1] = U[-2] + self.value * dx
        elif self.type == 'robin':
            if A is not None:
                if self.position == 'left':
                    A[0, 0] = self.coefficients[0] - self.coefficients[1] / dx
                    A[0, 1] = self.coefficients[1] / dx
                    F[0] = self.value
                elif self.position == 'right':
                    A[-1, -1] = self.coefficients[0] + self.coefficients[1] / dx
                    A[-1, -2] = -self.coefficients[1] / dx
                    F[-1] = self.value
            else:
                if self.position == 'left': #Explicit method with ghost node approach for robin boundary
                    U[0] = (self.value - self.coefficients[0] * U[0]) * dx / self.coefficients[1] + U[1]
                elif self.position == 'right':
                    U[-1] = (self.value - self.coefficients[0] * U[-1]) * dx / self.coefficients[1] + U[-2]
        else:
            raise ValueError("Unsupported boundary condition type. Choose 'dirichlet' or 'neumann'.")



def main():
#Example usage:
    boundary_conditions = [
        BoundaryCondition('left', 'neumann', 0, coefficients=None),
        BoundaryCondition('right', 'neumann', 0, coefficients=None)
    ]

    a = 0
    b = 6
    D = 0.01
    N = 50
    T = 100

    def source_term(t, x, U):
        #Q5: ((1-U)**2)*np.exp(-x)
        #BRATU: np.exp(2*U)
        return ((1-U)**2)*np.exp(-x)

    initial_condition = lambda x: np.zeros_like(x)
    dt_max = ((b-a)/N)**2/ (2 * D)
    dt = 0.5*dt_max
    def run_simulation():
        simulation = DiffusionSimulation(source_term,a, b, D, initial_condition, boundary_conditions, N, (0, T), 'explicit_euler', dt)
        x, t, U = simulation.solve()
        return U

    #cProfile.run('run_simulation()')

    print(f"Using dt={dt}")
    # start = time.time()
    simulation = DiffusionSimulation(source_term,a, b, D, initial_condition, boundary_conditions, N, (0, T), 'implicit_sparse', dt)

    x, t, U = simulation.solve()

    #end = time.time()
    #print(f"Time taken: {end-start}")
    #x, U_steady_state = simulation.solve_steady_state()


    selected_time_steps = [0, len(t) // 4, len(t) // 2, -1]  # Example: start, quarter, half, and final time

    plt.figure(figsize=(10, 6))
    for idx in selected_time_steps:
        plt.plot(x, U[idx], label=f't={t[idx]:.2f}')

    plt.xlabel('Spatial Domain (x)')
    plt.ylabel('Solution (U)')
    plt.title('Diffusion Solution at Selected Time Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()


