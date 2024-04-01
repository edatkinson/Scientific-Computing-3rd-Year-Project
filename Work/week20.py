# The code for the Implicit and Explicit Euler methods, along with the time and error calculations is below:

import numpy as np
from scipy.linalg import solve_banded
import time
import matplotlib.pyplot as plt

# Constants and initial conditions
D = 0.1  # Diffusion coefficient
L = 1.0  # Length of the domain
N = 100  # Number of grid points (101 points including boundary)
delta_x = L / N

x = np.linspace(0, L, N + 1)
u = np.sin(np.pi * x)  # Initial condition
exact_solution = np.exp(-0.2 * np.pi**2)  # Exact solution at T=2 and x=0.5

# Implicit Euler setup
delta_t_implicit = 0.1  # Time step size for Implicit Euler
alpha = D * delta_t_implicit / delta_x**2
A = np.zeros((3, N - 1))  # Coefficient matrix for Implicit Euler
A[0, 1:] = -alpha
A[1, :] = 1 + 2 * alpha
A[2, :-1] = -alpha

# Explicit Euler setup
# Adjusted time step size for stability in Explicit Euler
# Stability criterion for the explicit Euler method
delta_t_max_explicit = (delta_x ** 2) / (2 * D) #Ensures Stablity

beta = D * delta_t_max_explicit / delta_x**2
steps_to_T2_explicit = int(2 / delta_t_max_explicit)  # Number of steps to reach T=2 with new time step size

# Implicit Euler solver
def implicit_euler(u, A, N, steps):
    for _ in range(steps):
        b = u[1:-1]  # Right-hand side of the linear system
        u[1:-1] = solve_banded((1, 1), A, b)  # Solve the tridiagonal system
    return u

# Explicit Euler solver
def explicit_euler(u, beta, N, steps):
    for _ in range(steps):
        u_next = np.zeros_like(u)
        u_next[1:-1] = u[1:-1] + beta * (u[:-2] - 2*u[1:-1] + u[2:])
        u = u_next
    return u

# Time and error calculations for Implicit Euler
start_time = time.time()
u_implicit = implicit_euler(u.copy(), A, N, 20)
implicit_duration = time.time() - start_time
error_implicit = abs(u_implicit[N//2] - exact_solution)

# Time and error calculations for Explicit Euler
start_time = time.time()
u_explicit = explicit_euler(u.copy(), beta, N, steps_to_T2_explicit)
explicit_duration_adjusted = time.time() - start_time
error_explicit_adjusted = abs(u_explicit[N//2] - exact_solution)

# Outputs
print('Implicit Duration:',implicit_duration, 'Implicit Error:', error_implicit) 
print('Explicit Duration:',explicit_duration_adjusted, 'Explicit Error:',error_explicit_adjusted, steps_to_T2_explicit)


# Function to apply boundary conditions to the matrix A and vector b
def apply_boundary_conditions(A, b, u, boundary_conditions, delta_x, t, source_term):
    """
    Applies Dirichlet, Neumann, or Robin boundary conditions to the matrix A and vector b.

    Parameters:
        A (np.array): Tridiagonal matrix A for the implicit Euler method.
        b (np.array): Right-hand side vector b for the implicit Euler method.
        u (np.array): Current solution array.
        boundary_conditions (dict): Dictionary containing types and values for boundary conditions.
        delta_x (float): Spatial step size.
        t (float): Current time.
        source_term (callable): Function q(x, t, u) representing the source term.

    Returns:
        A, b: Modified matrix A and vector b after applying boundary conditions.
    """
    # Apply left boundary condition
    if boundary_conditions['left']['type'] == 'Dirichlet':
        b[0] = boundary_conditions['left']['value'](t)
    elif boundary_conditions['left']['type'] == 'Neumann':
        # Modify the first row of matrix A for Neumann boundary condition
        A[1, 0] += A[0, 1]  # Adjust main diagonal due to Neumann BC at the left boundary
        b[0] -= A[0, 1] * boundary_conditions['left']['value'](t) * delta_x
    elif boundary_conditions['left']['type'] == 'Robin':
        a, b_coeff, c = boundary_conditions['left']['value'](t)
        A[1, 0] += A[0, 1] * b_coeff * delta_x / (a * delta_x + b_coeff)
        b[0] -= A[0, 1] * c * delta_x / (a * delta_x + b_coeff)

    # Apply right boundary condition
    if boundary_conditions['right']['type'] == 'Dirichlet':
        b[-1] = boundary_conditions['right']['value'](t)
    elif boundary_conditions['right']['type'] == 'Neumann':
        # Modify the last row of matrix A for Neumann boundary condition
        A[1, -1] += A[2, -2]  # Adjust main diagonal due to Neumann BC at the right boundary
        b[-1] += A[2, -2] * boundary_conditions['right']['value'](t) * delta_x
    elif boundary_conditions['right']['type'] == 'Robin':
        a, b_coeff, c = boundary_conditions['right']['value'](t)
        A[1, -1] += A[2, -2] * b_coeff * delta_x / (a * delta_x + b_coeff)
        b[-1] += A[2, -2] * c * delta_x / (a * delta_x + b_coeff)

    # Add the source term to the right-hand side vector b
    for i in range(1, len(u)-1):
        b[i-1] += delta_t_implicit * source_term(x[i], t, u[i])

    return A, b

# Generalized implicit Euler solver
def generalized_implicit_euler(u, A, N, steps, boundary_conditions, source_term):
    """
    Solves the diffusion equation using a generalized implicit Euler method.

    Parameters:
        u (np.array): Initial condition array.
        A (np.array): Tridiagonal matrix A without considering boundary conditions.
        N (int): Number of spatial grid points.
        steps (int): Number of time steps to compute.
        boundary_conditions (dict): Dictionary containing types and values for boundary conditions.
        source_term (callable): Function q(x, t, u) representing the source term.

    Returns:
        np.array: Solution of u for the last time step.
    """
    for step in range(steps):
        # Time at the next step
        t = step * delta_t_implicit

        # Right-hand side of the linear system
        b = u[1:-1]

        # Apply boundary conditions to the matrix A and vector b
        A_mod, b_mod = apply_boundary_conditions(A.copy(), b.copy(), u, boundary_conditions, delta_x, t, source_term)

        # Solve the tridiagonal system
        u[1:-1] = solve_banded((1, 1), A_mod, b_mod)

        # Update the boundary values if they're time-dependent Dirichlet conditions
        if boundary_conditions['left']['type'] == 'Dirichlet' and callable(boundary_conditions['left']['value']):
            u[0] = boundary_conditions['left']['value'](t)
        if boundary_conditions['right']['type'] == 'Dirichlet' and callable(boundary_conditions['right']['value']):
            u[-1] = boundary_conditions['right']['value'](t)
    return u

# Example of how to use the generalized solver:
# Define the boundary conditions as a dictionary
boundary_conditions = {
    'left': {'type': 'Dirichlet', 'value': lambda t: np.sin(t)},  # Time-dependent Dirichlet condition at x=0
    'right': {'type': 'Neumann', 'value': lambda t: t}            # Homogeneous Neumann condition at x=L
}

# Define a sample source term function
source_term = lambda x, t, u: x  # No source term in this example

# Initial condition
u_initial = np.sin(np.pi * x)

num_time_steps = 101


u_final_generalized = generalized_implicit_euler(u_initial, A, N, num_time_steps, boundary_conditions, source_term)


# print(u_final_generalized, 'Generalized Implicit Euler Solution')

# Plot the final solution after the amount of time steps
plt.figure(figsize=(10, 6))
plt.plot(x, u_final_generalized, label='u(x) after 10 time steps')
plt.title('Solution of the Diffusion Equation with Source Term')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True)
plt.show()

