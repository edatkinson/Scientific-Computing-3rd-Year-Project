# Import necessary libraries
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
import matplotlib.pyplot as plt

# Constants
D = 1.0

# Function to set up the finite difference matrix for the second derivative
def setup_finite_difference_matrix(n_points, sigma, h, method='dense'):
    """
    Sets up the finite difference matrix for the second derivative.

    Parameters:
    - n_points (int): Number of grid points.
    - sigma (float): The parameter sigma in the equation.
    - h (float): The grid spacing.
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

    A[0, 0] = A[-1, -1] = 1
    A[0, 1] = A[-1, -2] = 0
    return A

# Function to setup the right-hand side of the equation
def setup_rhs(n_points, sigma, domain):
    """
    Sets up the right-hand side of the Poisson equation.

    Parameters:
    - n_points (int): Number of grid points.
    - sigma (float): The parameter sigma in the equation.
    - domain (ndarray): The domain of x values.

    Returns:
    - ndarray: The right-hand side vector.
    """
    return -(1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-domain**2 / (2 * sigma**2))

# Solve using dense matrix
def solve_dense(sigma, domain, h):
    """
    Solves the Poisson equation using a dense matrix approach.

    Parameters:
    - sigma (float): The parameter sigma in the equation.
    - domain (ndarray): The domain of x values.
    - h (float): The grid spacing.

    Returns:
    - ndarray: The solution vector u(x).
    """
    A_dense = setup_finite_difference_matrix(len(domain), sigma, h, method='dense')
    rhs = setup_rhs(len(domain), sigma, domain)
    rhs[0] = rhs[-1] = -1  # Apply boundary conditions
    u_dense = solve(A_dense, rhs)
    return u_dense

# Solve using sparse matrix
def solve_sparse(sigma, domain, h):
    """
    Solves the Poisson equation using a sparse matrix approach.

    Parameters:
    - sigma (float): The parameter sigma in the equation.
    - domain (ndarray): The domain of x values.
    - h (float): The grid spacing.

    Returns:
    - ndarray: The solution vector u(x).
    """
    A_sparse = setup_finite_difference_matrix(len(domain), sigma, h, method='sparse')
    rhs = setup_rhs(len(domain), sigma, domain)
    rhs[0] = rhs[-1] = -1 # Apply boundary conditions
    u_sparse = spsolve(A_sparse, rhs)
    return u_sparse

# Plotting function
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
    domain = np.linspace(-1, 1, no_points)  # 501 points in the domain
    h = domain[1] - domain[0]  # Step size
    u_dense = solve_dense(sigma=0.5, domain=domain, h=h)
    u_sparse = solve_sparse(sigma=0.1, domain=domain, h=h)
    plot_solutions(domain, u_dense, u_sparse)

if __name__ == '__main__':
    main()


