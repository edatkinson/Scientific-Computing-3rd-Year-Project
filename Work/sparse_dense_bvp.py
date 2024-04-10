# Import necessary libraries
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
import matplotlib.pyplot as plt
from reaction_diffusion import BoundaryCondition

# Constants
D = 1.0


def setup_finite_difference_matrix(n_points, h, equation_type, bc_left, bc_right, coefficients,method='sparse'):
    N = n_points

    main_diag = np.zeros(N)
    upper_diag = np.zeros(N - 1)
    lower_diag = np.zeros(N - 1)
    
    # Depending on the equation type, set up the diagonals
    if 'diffusion' in equation_type:
        D = coefficients.get('D')  
        main_diag[:] = -2 * D / h**2
        upper_diag[:] = lower_diag[:] = D / h**2 

    if 'convection' in equation_type:
        P = coefficients.get('P') 
        upper_diag += P / (2*h)
        lower_diag -= P / (2*h)

    if 'reaction' in equation_type:
        # For reaction terms, add to main diagonal based on the reaction coefficient
        # This part would be customized based on the specific reaction term
        R = coefficients.get('R')  # Reaction coefficient
        if R > 0:
            lower_diag += R / h  # Upwind scheme
        else:
            upper_diag -= R / h  # Upwind scheme
        
    

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
    - bc_left (Class object): The left boundary condition.
    - bc_right (Class object): The right boundary condition.
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

#Function to setup the right-hand side of the equation
def apply_rhs_boundary(rhs, h, bc_left, bc_right):

    if bc_left.bc == 'Dirichlet':
        rhs[0] = bc_left.value
    elif bc_left.bc == 'Neumann':
        rhs[0] = bc_left.value * h # forward difference
        
    if bc_right.bc == 'Dirichlet':
        rhs[-1] = bc_right.value
    elif bc_right.bc == 'Neumann':
        rhs[-1] =  bc_right.value *h
    
    return rhs

def setup_rhs_poisson(n_points, coefficients, h, bc_left, bc_right,domain):
    """
    Sets up the right-hand side of the Poisson equation.

    Parameters:
    - n_points (int): Number of grid points.
    - sigma (float): The parameter sigma in the equation.
    - domain (ndarray): The domain of x values.
    - h (float): The grid spacing.
    - bc_left (Class object): The left boundary condition.
    - bc_right (Class object): The right boundary condition.

    Returns:
    - ndarray: The right-hand side vector.
    """

    sigma = coefficients.get('sigma')  
    rhs = -(1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-domain**2 / (2 * sigma**2))
    #Apply boundary conditions
    rhs = apply_rhs_boundary(rhs, h, bc_left, bc_right)

    return rhs

def setup_rhs_reaction(n_points, coefficients, h, bc_left, bc_right, domain):
    R = coefficients.get('R')  # Reaction term
    rhs = np.ones(n_points) * R  # Reaction term
    rhs = apply_rhs_boundary(rhs, h, bc_left, bc_right)
    
    return rhs


def solve_dense(rhs,domain, h, bc_left, bc_right, coefficients, equation_type):
    """
    Solves the Poisson equation using a dense matrix approach. 

    Parameters:
        - sigma (float): The parameter in the equation.
        - domain (ndarray): The domain of x values.
        - h (float): The grid spacing.
        - bc_left (Class object): The left boundary condition.
        - bc_right (Class object): The right boundary condition.

    Returns:
        - ndarray: The solution vector u(x).
    """
    A_dense = setup_finite_difference_matrix(len(domain), h, equation_type, bc_left, bc_right,coefficients, method='dense')
    # rhs = setup_rhs(len(domain), sigma, domain,h, bc_left, bc_right)
    rhs_term = rhs(len(domain), coefficients,h, bc_left, bc_right, domain)
    u_dense = solve(A_dense, rhs_term)
    return u_dense

def solve_sparse(rhs, domain, h, bc_left, bc_right,coefficients, equation_type):
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
    A_sparse = setup_finite_difference_matrix(len(domain), h, equation_type, bc_left, bc_right, coefficients, method='sparse')
    #rhs = setup_rhs(len(domain), sigma, domain,h, bc_left, bc_right)
    rhs_term = rhs(len(domain), coefficients, h, bc_left, bc_right, domain)
    u_sparse = spsolve(A_sparse, rhs_term)
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
    equation_type_Q6 = 'convection-diffusion-reaction' #would need to set up a rhs function for this
    
    no_points = 81
    a = -1
    b = 1
    x = np.linspace(a, b, no_points)  # 501 points in the domain
    dx = x[1] - x[0]  # Step size
    dx = (b-a)/(no_points-1)

    bc_left = BoundaryCondition('Dirichlet', -1)
    bc_right = BoundaryCondition('Dirichlet', -1)


    coefficients_possion = {'D': 1.0, 'sigma': 0.5}
    #Poisson equation
    u_dense = solve_dense(setup_rhs_poisson,domain=x, h=dx,bc_left=bc_left,bc_right=bc_right, coefficients=coefficients_possion, equation_type='diffusion')
    u_sparse = solve_sparse(setup_rhs_poisson,domain=x, h=dx, bc_left=bc_left, bc_right=bc_right, coefficients=coefficients_possion, equation_type='diffusion')
    print(f"Max dense solution: {max(u_dense):5f}")
    print(f"Max sparse solution: {max(u_sparse):5f}")
    plot_solutions(x, u_dense, u_sparse)

    #P = 1
    coefficients_P1 = {'D': 1.0, 'P': 1, 'R': 1}
    u_dense = solve_dense(setup_rhs_reaction,domain=x, h=dx,bc_left=bc_left,bc_right=bc_right, coefficients=coefficients_P1, equation_type=equation_type_Q6)
    u_sparse = solve_sparse(setup_rhs_reaction,domain=x, h=dx, bc_left=bc_left, bc_right=bc_right, coefficients=coefficients_P1, equation_type=equation_type_Q6)
    print(f"Max dense solution: {max(u_dense):5f}")
    print(f"Max sparse solution: {max(u_sparse):5f}")
    plot_solutions(x, u_dense, u_sparse)

    #P = 10
    coefficients_P2 = {'D': 1.0, 'P': 10, 'R': 10}
    u_dense = solve_dense(setup_rhs_reaction,domain=x, h=dx,bc_left=bc_left,bc_right=bc_right, coefficients=coefficients_P2, equation_type=equation_type_Q6)
    u_sparse = solve_sparse(setup_rhs_reaction,domain=x, h=dx, bc_left=bc_left, bc_right=bc_right, coefficients=coefficients_P2, equation_type=equation_type_Q6)
    print(f"Max dense solution: {max(u_dense):5f}")
    print(f"Max sparse solution: {max(u_sparse):5f}")
    plot_solutions(x, u_dense, u_sparse)

    #P = 50
    coefficients_P3 = {'D': 1.0, 'P': 50, 'R': 50}
    u_dense = solve_dense(setup_rhs_reaction,domain=x, h=dx,bc_left=bc_left,bc_right=bc_right, coefficients=coefficients_P3, equation_type=equation_type_Q6)
    u_sparse = solve_sparse(setup_rhs_reaction,domain=x, h=dx, bc_left=bc_left, bc_right=bc_right, coefficients=coefficients_P3, equation_type=equation_type_Q6)
    print(f"Max dense solution: {max(u_dense):5f}")
    print(f"Max sparse solution: {max(u_sparse):5f}")
    plot_solutions(x, u_dense, u_sparse)

if __name__ == '__main__':
    main()


