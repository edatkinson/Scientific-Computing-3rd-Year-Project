import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from Diffusion_OO import BoundaryCondition
from scipy.integrate import solve_bvp


# Given parameters
D = 1  # Diffusion coefficient
sigma = 0.1  # Parameter sigma

# Define the differential equation system
def ode_system(x, y):
    return np.vstack((y[1], -1/(D * np.sqrt(2 * np.pi * sigma**2)) * np.exp(-x**2 / (2 * sigma**2))))

# Boundary conditions
def bc(ya, yb):
    return np.array([ya[0] + 1, yb[0] + 1])

# Initial mesh and initial guess
x = np.linspace(-1, 1, 100)
y_guess = np.zeros((2, x.size))
y_guess[0] = -1  # Initial guess for u

# Solve the boundary value problem
sol = solve_bvp(ode_system, bc, x, y_guess)

# Plot the solution
x_plot = np.linspace(-1, 1, 100)
y_plot = sol.sol(x_plot)[0]
plt.plot(x_plot, y_plot, label='u(x)')
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution of Poisson equation with scipy.solve_bvp')
plt.grid()
plt.show()

    
