# Import necessary libraries
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
import matplotlib.pyplot as plt
from Diffusion_OO import BoundaryCondition
from typing import Callable, Dict, Tuple
from scipy.optimize import root
from Equations_Functions import setup_rhs_poisson, setup_rhs_reaction

# Constants
D = 1.0

def setup_finite_difference_matrix(n_points, h, equation_type, bc_left, bc_right, coefficients,method='sparse'):
    N = n_points

    main_diag = np.zeros(N)
    upper_diag = np.zeros(N - 1)
    lower_diag = np.zeros(N - 1)
    
    # Depending on the equation type, set up the diagonals
    if 'diffusion' in equation_type:
        D = coefficients.get('D', 1)  
        main_diag[:] = -2 * D / h**2
        upper_diag[:] = lower_diag[:] = D / h**2 

    if 'convection' in equation_type:
        P = coefficients.get('P') 
        main_diag[1:] -= P / h
        lower_diag[:] += P / h
    
    # Create the sparse matrix
    if method == 'sparse':
        A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
    else:
        A = np.diag(main_diag) + np.diag(upper_diag, 1) + np.diag(lower_diag, -1)

    # Apply the boundary conditions
    A = apply_boundary_conditions(A, bc_left, bc_right, h)

    return A

def apply_boundary_conditions(A, bc_left, bc_right, h):
    """
    Applies the boundary conditions to the finite difference matrix A.

    Parameters:
    - A (ndarray or csr_matrix): The finite difference matrix.
    - bc_left (BoundaryCondition): The left boundary condition.
    - bc_right (BoundaryCondition): The right boundary condition.
    - h (float): The grid spacing.
    """

    # Apply left boundary condition
    if bc_left.type.lower() == 'dirichlet':
        A[0, 0] = 1.0
        A[0, 1] = 0.0
    elif bc_left.type.lower() == 'neumann':
        A[0, 0] = -1 / h
        A[0, 1] = 1 / h

    # Apply right boundary condition
    if bc_right.type.lower() == 'dirichlet':
        A[-1, -1] = 1.0
        A[-1, -2] = 0.0
    elif bc_right.type.lower() == 'neumann':
        A[-1, -2] = -1 / h
        A[-1, -1] = 1 / h

    return A

def apply_rhs_boundary(rhs, h, bc_left, bc_right):
    """
    Applies the boundary conditions to the right-hand side vector.

    Parameters:
    - rhs (ndarray): The right-hand side vector.
    - h (float): The grid spacing.
    - bc_left (BoundaryCondition): The left boundary condition.
    - bc_right (BoundaryCondition): The right boundary condition.
    """

    # Apply left boundary condition
    if bc_left.type.lower() == 'dirichlet':
        rhs[0] = bc_left.value
    elif bc_left.type.lower() == 'neumann':
        rhs[0] += bc_left.value * h  # Modify the first entry accordingly

    # Apply right boundary condition
    if bc_right.type.lower() == 'dirichlet':
        rhs[-1] = bc_right.value
    elif bc_right.type.lower() == 'neumann':
        rhs[-1] -= bc_right.value * h  # Modify the last entry accordingly

    return rhs

def solve_dense(rhs,domain, h, bc_left, bc_right, coefficients, equation_type):
    """
    Parameters:
        - rhs (Callable): Function to setup the right-hand side of the equation.
        - domain (ndarray): The domain of x values.
        - h (float): The grid spacing.
        - bc_left (BoundaryCondition): The type ('Dirichlet' or 'Neumann') and value of the left boundary condition.
        - bc_right (BoundaryCondition): The type ('Dirichlet' or 'Neumann') and value of the right boundary condition.
        - coefficients (dict): Dictionary of coefficients used in the differential equation.
        - equation_type (str): Type of the equation to solve.
     Returns:
        - ndarray: The solution vector u(x).
    """
    A_dense = setup_finite_difference_matrix(len(domain), h, equation_type, bc_left, bc_right,coefficients, method='dense')
    rhs_term = apply_rhs_boundary(rhs(len(domain), coefficients, domain), h, bc_left, bc_right)

    u_dense = solve(A_dense, rhs_term)
    return u_dense

def finite_difference_scheme(N,a,b,h, D, q_func, bc_left, bc_right):
    def system(u, p):
        # Interior points
        du2dx2 = (u[:-2] - 2*u[1:-1] + u[2:]) / h**2
        # Evaluate q at interior points
        x_inner = np.linspace(a + h, b - h, N-2)
        q_term = q_func(u[1:-1], x_inner, p)
        F = D * du2dx2 - q_term

        # Apply boundary conditions
        F = np.concatenate(([0], F, [0]))  # Start with Dirichlet BCs as placeholders
        
        # Dirichlet conditions
        F[0] = u[0] - bc_left.value
        F[-1] = u[-1] - bc_right.value

        # Neumann conditions at the boundaries 
        if bc_left.type == 'Neumann':
            F[0] = (u[1] - u[0]) / h - bc_left.value
        if bc_right.type == 'Neumann':
            F[-1] = (u[-1] - u[-2]) / h - bc_right.value

        #Add in Robin conditions

        return F
    
    return system

# Function that represents the discretized system of equations
def solve_bvp_root(N, a, b, D, q_func, bc_left, bc_right, p):
    # Initial guess for the solution
    x = np.linspace(a, b, N)
    u_initial = np.zeros(N)
    
    # Define step size
    h = (b - a) / (N - 1)
    
    # Get the finite difference scheme for the system
    system = finite_difference_scheme(N,a,b, h, D, q_func, bc_left, bc_right)
    
    # Use scipy.optimize.root to solve the system
    sol = root(system, u_initial, args=(p,), method='hybr')

    return x, sol.x

# Define the function q(x, u; µ) as needed by the user
def q_func(u,x, p):
    sigma = p
    rhs = -(1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-x**2 / (2 * sigma**2))
    return rhs



def solve_sparse(rhs, domain, h, bc_left, bc_right,coefficients, equation_type):
    """

    Parameters:
        - rhs (Callable): Function to setup the right-hand side of the equation.
        - domain (ndarray): The domain of x values.
        - h (float): The grid spacing.
        - bc_left (BoundaryCondition): The type ('Dirichlet' or 'Neumann') and value of the left boundary condition.
        - bc_right (BoundaryCondition): The type ('Dirichlet' or 'Neumann') and value of the right boundary condition.
        - coefficients (dict): Dictionary of coefficients used in the differential equation.
        - equation_type (str): Type of the equation to solve.
     Returns:
        - ndarray: The solution vector u(x).

     """
    A_sparse = setup_finite_difference_matrix(len(domain), h, equation_type, bc_left, bc_right, coefficients, method='sparse')
    rhs_term = apply_rhs_boundary(rhs(len(domain), coefficients,domain), h, bc_left, bc_right)

    u_sparse = spsolve(A_sparse, rhs_term)
    return u_sparse


def plot_solutions(domain, u_dense, u_sparse, sigma, name):
    plt.figure(figsize=(10, 6))
    plt.plot(domain, u_dense, label=f'Dense matrix solution, parameter = {sigma}')
    plt.plot(domain, u_sparse, label=f'Sparse matrix solution, parameter = {sigma}')
    plt.title(f'Solutions of the {name} Equation')
    plt.xlabel('Domain x')
    plt.ylabel('Solution u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

def solve_equation(storage_type: str, rhs: Callable, domain: np.ndarray, h: float, bc_left, bc_right, coefficients: Dict[str, float], equation_type: str):
    """
    Solves a differential equation using either a dense or sparse matrix approach based on the user's choice.

    Args:
    storage_type (str): 'dense' or 'sparse', indicating the type of matrix storage and solver to use.
    rhs (Callable): Function to setup the right-hand side of the equation.
    domain (np.ndarray): The domain over which the equation is solved.
    h (float): The step size in the domain.
    bc_left (BoundaryCondition): Boundary condition at the left end of the domain.
    bc_right (BoundaryCondition): Boundary condition at the right end of the domain.
    coefficients (Dict[str, float]): Dictionary of coefficients used in the differential equation.
    equation_type (str): Type of the equation to solve (e.g., 'diffusion', 'convection', 'diffusion-convection').

    Returns:
    np.ndarray: The solution to the differential equation.
    """
    if storage_type == 'sparse':
        return solve_sparse(rhs, domain, h, bc_left, bc_right, coefficients, equation_type)
    elif storage_type == 'dense':
        return solve_dense(rhs, domain, h, bc_left, bc_right, coefficients, equation_type)
    else:
        raise ValueError(f"Unsupported storage type '{storage_type}'. Choose 'dense' or 'sparse'.")


def main():
    equation_type_Q6 = 'convection-diffusion-reaction' #would need to set up a rhs function for this
    
    no_points = 151
    a = -1
    b = 1
    x = np.linspace(a, b, no_points)  # 501 points in the domain
    dx = x[1] - x[0]  # Step size
    dx = (b-a)/(no_points-1)
    D = 1
    p = 0.5

    boundary_conditions = [
    BoundaryCondition('left', 'dirichlet', -1, coefficients=None),
    BoundaryCondition('right', 'dirichlet', -1, coefficients=None)
]
    
    bc_left = boundary_conditions[0]
    bc_right = boundary_conditions[1]

    x, u = solve_bvp_root(no_points, a, b, D, q_func, bc_left, bc_right, p)
    plt.plot(x, u, label='Solution to the Poisson Eq')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution of the BVP')    
    plt.grid(True)
    plt.show()


    coefficients_possion = {'D': 1.0, 'sigma': 0.5}
    #Poisson equation
    u_dense = solve_dense(setup_rhs_poisson,domain=x, h=dx,bc_left=bc_left,bc_right=bc_right, coefficients=coefficients_possion, equation_type='diffusion')
    u_sparse = solve_sparse(setup_rhs_poisson,domain=x, h=dx, bc_left=bc_left, bc_right=bc_right, coefficients=coefficients_possion, equation_type='diffusion')
    print(f"Max dense solution: {max(u_dense):5f}")
    print(f"Max sparse solution: {max(u_sparse):5f}")
    plot_solutions(x, u_dense, u_sparse, coefficients_possion.get('sigma'), 'Poisson')

    #P = 1
    coefficients_P1 = {'D': 1.0, 'P': 1}
    u_dense = solve_dense(setup_rhs_reaction,domain=x, h=dx,bc_left=bc_left,bc_right=bc_right, coefficients=coefficients_P1, equation_type=equation_type_Q6)
    u_sparse = solve_sparse(setup_rhs_reaction,domain=x, h=dx, bc_left=bc_left, bc_right=bc_right, coefficients=coefficients_P1, equation_type=equation_type_Q6)
    print(f"Max dense solution: {max(u_dense):5f}")
    print(f"Max sparse solution: {max(u_sparse):5f}")
    plot_solutions(x, u_dense, u_sparse, coefficients_P1.get('P'), 'Reaction-Convection-Diffusion')

    #P = 10
    coefficients_P2 = {'D': 1.0, 'P': 10}
    u_dense = solve_dense(setup_rhs_reaction,domain=x, h=dx,bc_left=bc_left,bc_right=bc_right, coefficients=coefficients_P2, equation_type=equation_type_Q6)
    u_sparse = solve_sparse(setup_rhs_reaction,domain=x, h=dx, bc_left=bc_left, bc_right=bc_right, coefficients=coefficients_P2, equation_type=equation_type_Q6)
    print(f"Max dense solution: {max(u_dense):5f}")
    print(f"Max sparse solution: {max(u_sparse):5f}")
    plot_solutions(x, u_dense, u_sparse, coefficients_P2.get('P'), 'Reaction-Convection-Diffusion')

    #P = 50
    coefficients_P3 = {'D': 1.0, 'P': 50}
    u_dense = solve_dense(setup_rhs_reaction,domain=x, h=dx,bc_left=bc_left,bc_right=bc_right, coefficients=coefficients_P3, equation_type=equation_type_Q6)
    u_sparse = solve_sparse(setup_rhs_reaction,domain=x, h=dx, bc_left=bc_left, bc_right=bc_right, coefficients=coefficients_P3, equation_type=equation_type_Q6)
    print(f"Max dense solution: {max(u_dense):5f}")
    print(f"Max sparse solution: {max(u_sparse):5f}")
    plot_solutions(x, u_dense, u_sparse, coefficients_P3.get('P'), 'Reaction-Convection-Diffusion')


if __name__ == '__main__':
    main()
