import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from reaction_diffusion import BoundaryCondition


def finite_difference_scheme(N, h, D, q_func, bc_left, bc_right):
    def system(u, p):
        # Interior points
        du2dx2 = (u[:-2] - 2*u[1:-1] + u[2:]) / h**2
        # Evaluate q at interior points
        x_inner = np.linspace(a + h, b - h, N-2)
        q_term = q_func(u[1:-1], x_inner, p)
        F = D * du2dx2 + q_term

        # Apply boundary conditions
        F = np.concatenate(([0], F, [0]))  # Start with Dirichlet BCs as placeholders
        
        # Dirichlet conditions
        F[0] = u[0] - bc_left.value
        F[-1] = u[-1] - bc_right.value

        # Neumann conditions at the boundaries 
        if bc_left.bc == 'Neumann':
            F[0] = (u[1] - u[0]) / h - bc_left.value
        if bc_right.bc == 'Neumann':
            F[-1] = (u[-1] - u[-2]) / h - bc_right.value
        
        #Add in Robin conditions

        return F
    
    return system

# Function that represents the discretized system of equations
def solve_bvp(N, a, b, D, q_func, bc_left, bc_right, p):
    # Initial guess for the solution
    x = np.linspace(a, b, N)
    u_initial = np.zeros(N)
    
    # Define step size
    h = (b - a) / (N - 1)
    
    # Get the finite difference scheme for the system
    system = finite_difference_scheme(N, h, D, q_func, bc_left, bc_right)
    
    # Use scipy.optimize.root to solve the system
    sol = root(system, u_initial, args=(p,), method='hybr')

    return x, sol.x

# Define the function q(x, u; µ) as needed by the user
def q_func(u,x, p):
    # The user needs to define this function based on their specific problem
    # This is a placeholder for demonstration purposes
    sigma = p
    rhs = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-x**2 / (2 * sigma**2))
    return rhs

if __name__ == "__main__":
    # Boundary conditions example: Left = Dirichlet, Right = Neumann
    bc_left = BoundaryCondition('Dirichlet', -1)
    bc_right = BoundaryCondition('Dirichlet', -1)

    
    # Domain and parameters
    a, b = 0, 1  # Domain from a to b
    N = 51  # Number of discretization points
    D = 1.0  # Diffusion coefficient
    p = 0.5  # Parameter sigma

    # Solve the BVP
    x, u = solve_bvp(N, a, b, D, q_func, bc_left, bc_right, p)
    
    # Plot the solution
    plt.plot(x, u, label='Finite Difference Solution')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution of the BVP')
    
    # u = (bc_right['value'] - bc_left['value']) / (b - a) * (x-a) + bc_left['value']
    # plt.plot(x, u)
    #u = -1/2*D * (x-a)*(x-b) + (bc_right['value'] - bc_left['value']) / (b - a) * (x-a) + bc_left['value']
    plt.legend()
    plt.show()


    
