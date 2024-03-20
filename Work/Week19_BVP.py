import numpy as np 
import matplotlib.pyplot as plt


def u(x):
    return np.exp(x)

def u_prime(x):
    return 3*x**2


def forwards_diff(f, a, delta_x):
    u_a = f(a)
    u_b = f(a + delta_x)
    u_prime = (u_b - u_a) / delta_x
    return u_prime

def backwards_diff(f, a, delta_x):
    u_a = f(a)
    u_b = f(a - delta_x)
    u_prime = (u_a - u_b) / delta_x
    return u_prime

def central_diff(f, a, delta_x):
    u_a = f(a)
    u_b = f(a + delta_x)
    u_c = f(a - delta_x)
    u_prime = (u_b - u_c) / (2*delta_x)
    return u_prime


def second_order_central(f,a, delta_x):
    u_a = f(a)
    u_b = f(a + delta_x)
    u_c = f(a - delta_x)
    u_prime = (u_b - 2*u_a + u_c) / (delta_x**2)
    return u_prime

def q(x):
    return 1

N = 50
a, b = 0, 2  # Domain
alpha, beta = 1, 2  # Dirichlet Boundary Conditions

dx = (b - a) / N  # Grid spacing

D = 1/4

# Points including boundaries
x = np.linspace(a, b, N+1)

# Initialize A matrix and b vector for the interior points only
A = np.zeros((N-1, N-1))
B = np.zeros(N-1)  # Right-hand side vector

# Constructing the matrix A for interior points
# for i in range(1,N):
#     A[i-1, i-1] = -2 * D - (dx**2) * q(x[i])
#     if i > 1:
#         A[i-1, i-2] = D  # Lower diagonal
#     if i < N-1:
#         A[i-1, i] = D # Upper diagonal

#Constructing A using np.fill_diagonal:
np.fill_diagonal(A, -2*D - (dx**2)*q(x))
np.fill_diagonal(A[1:], D)
np.fill_diagonal(A[:, 1:], D)

A = A / dx**2  # Apply the scaling factor


B[0] -= D * alpha / dx**2
B[-1] -= D * beta / dx**2
# Solve the system for the interior points
u_internal = np.linalg.solve(A, B)

# Construct the full solution vector including the boundary conditions
u = np.zeros(N+1)  # Full solution, including boundaries
u[0] = alpha  # Apply the left boundary condition
u[-1] = beta  # Apply the right boundary condition
u[1:N] = u_internal  # Insert the solution for the interior points


u = 1/(-2*D) * (x-a)*(x-b) + ((beta-alpha)/(b-a)) * (x-a) + alpha
plt.plot(x,u, '-o', label='Analytical Solution')

# Plotting
plt.plot(x, u, '-o', label='Numerical Solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution of 1D Laplace Equation with Dirichlet BCs')
plt.legend()
plt.show()








