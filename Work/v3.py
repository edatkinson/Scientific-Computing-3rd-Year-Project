import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from matplotlib.animation import FuncAnimation



def generalized_explicit_euler(u0, D, x, delta_x, delta_t, steps, boundary_conditions, source_term,mu):
    N = len(u0)
    u = u0.copy()
    alpha = D * delta_t / delta_x**2
    solutions = [u.copy()]
    for step in range(steps):
        t = step * delta_t
        # Vectorized update for the interior points
        u[1:-1] = u[1:-1] + alpha * (u[:-2] - 2*u[1:-1] + u[2:]) + delta_t * source_term(x[1:-1], t, u[1:-1],mu)
        # Apply boundary conditions
        u[0] = boundary_conditions['left']['value'](t)
        u[-1] = boundary_conditions['right']['value'](t)
        solutions.append(u.copy())
    return u, solutions

def generalized_implicit_euler(u0, D, x, delta_x, delta_t, steps, boundary_conditions, source_term,mu):
    N = len(u0)
    # Correct setup for banded matrix A with shape (3, N)
    A = np.zeros((3, N))  # Adjust size for off-diagonals and main diagonal
    alpha = D * delta_t / delta_x**2

    # Setup A for banded solver, paying attention to boundary conditions
    A[0, 2:] = -alpha  # Upper diagonal
    A[1, :] = 1 + 2*alpha  # Main diagonal, including first and last for Dirichlet BC
    A[2, :-2] = -alpha  # Lower diagonal

    u = u0.copy()
    solutions = [u.copy()]
    for step in range(steps):
        t = step * delta_t
        b = u.copy()  # Copy the previous time step's solution into b
        # Adjust b for source term in interior points
        b[1:-1] += delta_t * source_term(x[1:-1], t, u[1:-1],mu)
        
        # Apply Dirichlet boundary conditions directly to b, and adjust A if necessary
        b[0] = boundary_conditions['left']['value'](t)
        b[-1] = boundary_conditions['right']['value'](t)
        
        # Solve using banded solver
        u = solve_banded((1, 1), A, b)
        solutions.append(u.copy())
    return u, solutions



def animate_solutions(x, solutions_explicit, solutions_implicit):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    axs[0].set_title('Explicit Euler Solution')
    axs[1].set_title('Implicit Euler Solution')
    line_explicit, = axs[0].plot(x, solutions_explicit[0], color='blue')
    line_implicit, = axs[1].plot(x, solutions_implicit[0], color='red')
    
    def init():
        line_explicit.set_ydata([np.nan] * len(x))
        line_implicit.set_ydata([np.nan] * len(x))
        return line_explicit, line_implicit
    
    def animate(i):
        line_explicit.set_ydata(solutions_explicit[i])
        line_implicit.set_ydata(solutions_implicit[i])
        return line_explicit, line_implicit
    
    ani = FuncAnimation(fig, animate, frames=len(solutions_explicit), init_func=init, blit=True, interval=50)
    plt.show()

def run_simulation():
    # Constants and initial conditions
    D = 1  # Diffusion coefficient
    L = 2.0  # Length of the domain
    T = 1  # Total time to simulate
    N = 101  # Number of spatial grid points
    delta_x = L / (N - 1)
    x = np.linspace(0, L, N)
    mu = 0.5
    a = 0
    b = L
    

    # Example initial condition: sin(2*pi*x)
    initial_condition = lambda x: 0.5 * x*(2-x) + np.sin(np.pi*x)
    u0 = initial_condition(x)

    # Boundary conditions
    boundary_conditions = {
        'left': {'type': 'Dirichlet', 'value': lambda t: 1},
        'right': {'type': 'Dirichlet', 'value': lambda t: 1}
    }

    # Example source term: zero for simplicity
    source_term = lambda x, t, u, mu: 0*x

    # Time step calculation based on stability criterion for explicit method
    delta_t_max_explicit = (delta_x ** 2) / (4 * D)
    print(delta_t_max_explicit)
    steps_explicit = int(T / delta_t_max_explicit) + 1
    delta_t_explicit = T / (steps_explicit - 1)

    # Vary the number of time steps for the implicit method to get a more accurate solution
    steps_implicit = 500
    delta_t_implicit = T / steps_implicit


    u,solutions_exp = generalized_explicit_euler(u0, D, x, delta_x, delta_t_explicit, steps_explicit, boundary_conditions, source_term,mu)
    u,solutions_imp = generalized_implicit_euler(u0, D, x, delta_x, delta_t_implicit, steps_implicit, boundary_conditions, source_term,mu)
    animate_solutions(x, solutions_exp, solutions_imp)



# Ensure necessary libraries are imported before calling run_simulation
if __name__ == '__main__':
    run_simulation()


