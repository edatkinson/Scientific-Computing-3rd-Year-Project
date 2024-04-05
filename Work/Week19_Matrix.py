
# Adjusting the code to use scipy.optimize.root function to solve the nonlinear Poisson equation.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.sparse import diags

# Setting up the grid
D =1
a = 0
b = 1
N = 500
alpha, beta = 0,0
dx = (b-a) / N
x = np.linspace(a, b, N+1)
par = [0.1]

def construct_matrix_A(N, dx):
    # Create a tri-diagonal matrix with -2 on the diagonal and 1 on the off-diagonals
    D = 1
    A = diags([1, -2, 1], [-1, 0, 1], shape=(N-1, N-1)) *D/ dx**2
    return A.tocsr()

def F(u, alpha, beta, dx, x,par):
    # Function representing the discretized nonlinear Poisson equation.
    N = len(u) + 1
    u_full = np.concatenate(([alpha], u, [beta]))
    # Use the matrix A to apply the finite difference method
    Au = construct_matrix_A(N, dx).dot(u_full[1:-1])
    # Add the nonlinear term with x-dependency
    return Au + q(u_full[1:-1], x[1:-1],par)

def q(u, x,par):
    # Nonlinear term q(u, x).
    mu = par[0]
    return np.exp(u*mu*x)

# Define the initial guess for the root finding as ones
# u_guess = np.array([1 / (-2 * D) * (xi - a) * (xi - b) + 
#                 ((beta - alpha) / (b - a)) * (xi - a) + alpha for xi in x[1:-1]])

u_guess = np.ones(N-1)

# Use scipy.optimize.root to solve the nonlinear system, including x in the arguments for F
sol = root(F, u_guess, args=(alpha, beta, dx, x,par))

# Extract the solution
u_sol = sol.x

# Add the boundary points to the solution
u_sol_full = np.concatenate(([alpha], u_sol, [beta]))

# Plot the solution
plt.plot(x, u_sol_full, label='Numerical solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()





