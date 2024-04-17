import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve
import matplotlib.pyplot as plt
from scipy.optimize import root, fsolve
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
import time
import cProfile


class DiffusionSimulation:
    def __init__(self, source_term, a, b, D, initial_condition, boundary_conditions, N, time_span, method, dt=None):
        self.a = a
        self.b = b
        self.D = D
        self.initial_condition = initial_condition
        self.boundary_conditions = boundary_conditions
        self.N = N
        self.dx = (b - a) / (N-1)
        self.time_span = time_span
        self.method = method
        self.dt = dt if dt is not None else (time_span[1] - time_span[0]) / 100  # default time step for implicit 
        self.source_term = source_term

    def diffusion_rhs(self,t, U):
        # Initialize the derivative vector
        dUdt = np.zeros_like(U)
        x = np.linspace(self.a, self.b, len(U))
        # Apply central difference for interior points using vector methods instead of loops
        dUdt[1:-1] = self.D * (U[2:] - 2*U[1:-1] + U[:-2]) / self.dx**2 + self.source_term(t, x[1:-1], U[1:-1])
        return dUdt

    def apply_boundary_conditions(self, U, A=None, F=None):
        # Apply boundary conditions
        for bc in self.boundary_conditions:
            bc.apply(U, A, F, self.dx, self.D, self.dt)

    def explicit_euler_step(self, t,U):
        dUdt = self.diffusion_rhs(t,U)
        U_new = U + self.dt * dUdt
        self.apply_boundary_conditions(U_new)
        return U_new

    def implicit_euler_step(self, t,U, storage='dense'):
        N = len(U)
        A = np.eye(N) - self.dt * self.D * (np.diag(np.ones(N-1), -1) - 2*np.diag(np.ones(N), 0) + np.diag(np.ones(N-1), 1)) / self.dx**2
        F = U.copy() + self.dt *self.source_term(t, np.linspace(self.a, self.b, N), U) 
        self.apply_boundary_conditions(U, A, F)
        if storage == 'dense':
            U_new = solve(A, F)
        else: 
            A = sp.csr_matrix(A) #
            U_new = spsolve(A, F)

        return U_new
    
    def diffusion_solver(self, method):
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
            else:
                # Compute the source term implicitly
                source_term_func = lambda U_next: self.source_term(n * self.dt, x, U_next)

            U_next = self.solve_u_next(U_prev, source_term_func, x, n)
            u[n, :] = U_next

        return x, u

    def solve_u_next(self, U_prev, source_term_func, x, n):
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
            for i in range(timesteps):
                if self.method == 'explicit_euler':
                    if self.dt > self.dx**2 / (2 * self.D):
                        raise ValueError(f"Explicit Euler method is unstable for dt > dx^2 / (2 * D). Current dt={self.dt} and dx={self.dx}.")
                    U = self.explicit_euler_step(current_t, U)
                elif self.method == 'implicit_euler_dense':
                    U = self.implicit_euler_step(current_t, U, storage = 'dense')
                elif self.method == 'implicit_euler_sparse':
                    U = self.implicit_euler_step(current_t, U, storage = 'sparse')
                else:
                    raise ValueError("Unsupported method. Choose 'explicit_euler', 'implicit_euler_dense' or 'implicit_euler_sparse.")
                U_sol.append(U.copy())
                current_t += self.dt
            return x, np.linspace(self.time_span[0], self.time_span[1], timesteps+1), np.array(U_sol)

class BoundaryCondition:
    def __init__(self, position, type, value, coefficients):
        self.position = position
        self.type = type
        self.value = value
        self.coefficients = coefficients #coefficients for Robin boundary condition (alpha, beta)

    def apply(self, U, A=None, F=None, dx=None, D=None, dt=None):
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
        simulation = DiffusionSimulation(source_term,a, b, D, initial_condition, boundary_conditions, N, (0, T), method='explicit_euler', dt=dt)
        x, t, U = simulation.solve()
        return U

    #cProfile.run('run_simulation()')

    print(f"Using dt={dt}")
    # start = time.time()
    simulation = DiffusionSimulation(source_term,a, b, D, initial_condition, boundary_conditions, N, (0, T), method='IMEX', dt=dt)

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


