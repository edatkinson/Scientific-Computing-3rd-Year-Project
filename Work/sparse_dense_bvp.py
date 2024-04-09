# Import necessary libraries
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
import matplotlib.pyplot as plt
from reaction_diffusion import BoundaryCondition

# Constants
D = 1.0

# Function to set up the finite difference matrix for the second derivative
def setup_finite_difference_matrix(n_points, sigma, h, bc_left, bc_right,method='dense'):
    """
    Sets up the finite difference matrix for the second derivative.

    Parameters:
    - n_points (int): Number of grid points.
    - sigma (float): The parameter sigma in the equation.
    - h (float): The grid spacing.
    - bc_left (tuple): The type ('Dirichlet' or 'Neumann') and value of the left boundary condition.
    - bc_right (tuple): The type ('Dirichlet' or 'Neumann') and value of the right boundary condition.
    - method (str): 'dense' for a dense matrix, 'sparse' for a sparse matrix.

    Returns:
    - ndarray or csr_matrix: The finite difference matrix.
    """
    diagonal = -2.0 * np.ones(n_points)
    off_diagonal = np.ones(n_points - 1)
    if method == 'dense':
        A = (np.diag(diagonal, 0) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)) * D / h**2
    elif method == 'sparse':
        A = diags([diagonal, off_diagonal, off_diagonal], [0, 1, -1], format='csr') * D / h**2

    A = apply_boundary_conditions(A, bc_left, bc_right, h)

    return A

def apply_boundary_conditions(A, bc_left, bc_right, h):
    """
    Applies the boundary conditions to the finite difference matrix A.

    Parameters:
    - A (ndarray or csr_matrix): The finite difference matrix.
    - bc_left (tuple): The type ('Dirichlet' or 'Neumann') and value of the left boundary condition.
    - bc_right (tuple): The type ('Dirichlet' or 'Neumann') and value of the right boundary condition.
    - h (float): The grid spacing.
    """
    
    

    if bc_left.bc == 'Dirichlet':
        A[0, 0] = 1.0
        A[0, 1] = 0.0
    elif bc_left.bc == 'Neumann':
        A[0,0] = -1/h
        A[0,1] = 1/h

    if bc_right.bc == 'Dirichlet':
        A[-1, -1] = 1.0
        A[-1, -2] = 0.0
    elif bc_right.bc == 'Neumann':
        A[-1,-2] = -1/h
        A[-1,-1] = 1/h

    return A

# Function to setup the right-hand side of the equation
def setup_rhs(n_points, sigma, domain, h, bc_left, bc_right):
    """
    Sets up the right-hand side of the Poisson equation.

    Parameters:
    - n_points (int): Number of grid points.
    - sigma (float): The parameter sigma in the equation.
    - domain (ndarray): The domain of x values.
    - h (float): The grid spacing.
    - bc_left (tuple): The type ('Dirichlet' or 'Neumann') and value of the left boundary condition.
    - bc_right (tuple): The type ('Dirichlet' or 'Neumann') and value of the right boundary condition.

    Returns:
    - ndarray: The right-hand side vector.
    """
    rhs = -(1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-domain**2 / (2 * sigma**2))

    
    if bc_left.bc == 'Dirichlet':
        rhs[0] = bc_left.value
    elif bc_left.bc == 'Neumann':
        rhs[0] = bc_left.value * h # forward difference
        

    if bc_right.bc == 'Dirichlet':
        rhs[-1] = bc_right.value
    elif bc_right.bc == 'Neumann':
        rhs[-1] =  bc_right.value *h
        

    return rhs


def solve_dense(sigma, domain, h, bc_left, bc_right):
    """
    Solves the Poisson equation using a dense matrix approach. 

    Parameters:
        - sigma (float): The parameter sigma in the equation.
        - domain (ndarray): The domain of x values.
        - h (float): The grid spacing.
        - bc_left (tuple): The type ('Dirichlet' or 'Neumann') and value of the left boundary condition.
        - bc_right (tuple): The type ('Dirichlet' or 'Neumann') and value of the right boundary condition.

    Returns:
        - ndarray: The solution vector u(x).
    """

    A_dense = setup_finite_difference_matrix(len(domain), sigma, h, bc_left, bc_right, method='dense')
    rhs = setup_rhs(len(domain), sigma, domain,h, bc_left, bc_right)
    u_dense = solve(A_dense, rhs)
    return u_dense

def solve_sparse(sigma, domain, h, bc_left, bc_right):
    """
    Solves the Poisson equation using a sparse matrix approach.

    Parameters:
        - sigma (float): The parameter sigma in the equation.
        - domain (ndarray): The domain of x values.
        - h (float): The grid spacing.
        - bc_left (tuple): The type ('Dirichlet' or 'Neumann') and value of the left boundary condition.
        - bc_right (tuple): The type ('Dirichlet' or 'Neumann') and value of the right boundary condition.
     Returns:
        - ndarray: The solution vector u(x).

     """
    A_sparse = setup_finite_difference_matrix(len(domain), sigma, h, bc_left, bc_right, method='sparse')
    rhs = setup_rhs(len(domain), sigma, domain,h, bc_left, bc_right)
    u_sparse = spsolve(A_sparse, rhs)
    return u_sparse


def plot_solutions(domain, u_dense, u_sparse):
    plt.figure(figsize=(10, 6))
    plt.plot(domain, u_dense, label='Dense matrix solution, sigma=0.5')
    plt.plot(domain, u_sparse, label='Sparse matrix solution, sigma=0.1')
    plt.title('Solutions of the Poisson Equation')
    plt.xlabel('Domain x')
    plt.ylabel('Solution u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()



def main():
    no_points = 51
    a = -1
    b = 1
    x = np.linspace(a, b, no_points)  # 501 points in the domain
    dx = x[1] - x[0]  # Step size
    bc_left = BoundaryCondition('Dirichlet', -1)
    bc_right = BoundaryCondition('Dirichlet', -1)
    u_dense = solve_dense(sigma=0.5, domain=x, h=dx,bc_left=bc_left,bc_right=bc_right)
    u_sparse = solve_sparse(sigma=0.1, domain=x, h=dx, bc_left=bc_left, bc_right=bc_right)
    plot_solutions(x, u_dense, u_sparse)


if __name__ == '__main__':
    main()


