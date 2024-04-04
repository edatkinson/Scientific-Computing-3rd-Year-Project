
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

#Progression:
#Week 19: Solve 2nd Order Differential Equation with Finite Difference Method with a source term q(x) or q(u)
#Use Numpy or Scipy to solve the system of equations

N = 100
a = 0
b = 1
x = np.linspace(a,b,N+1)
alpha, beta = 0,0
dx = (b-a)/N
x_int = x[1:-1]

def q(x):
    return np.ones(np.size(x)) #np.sin(x**2)

def Dirichlet_BC(u,N,dx,alpha,beta,q,x_int):
    F = np.zeros(N-1)
    q_values = q(x_int) 

    F[0] = (u[1] - 2*u[0] + alpha)/dx**2 + q_values[0]

    for i in range(1,N-2):
        F[i] = (u[i+1] - 2*u[i] + u[i-1])/dx**2  + q_values[i]
    
    F[N-2] = (beta - 2*u[N-2] + u[N-3])/dx**2  + q_values[N-2]

    return F 

def Neumann_BC(u, N, dx, alpha, beta, q, x_int):
    F = np.zeros(N-1)
    q_values = q(x_int) 

    # Dirichlet condition at the left boundary
    F[0] = (u[1] - 2*u[0] + alpha)/dx**2 + q_values[0]

    # Interior points
    for i in range(1, N-2):
        F[i] = (u[i+1] - 2*u[i] + u[i-1])/dx**2  + q_values[i]
    
    # Neumann condition at the right boundary
    # Approximating u'(b) = beta with a backward difference: (u[N-1] - u[N-2]) / dx = beta
    # Rearrange to express u[N-1] in terms of u[N-2] and beta, then substitute in the discretized equation
    F[N-2] = (u[N-2] - 2*u[N-2] + u[N-3])/dx**2 + 2*beta/dx + q_values[N-2]

    return F

def Robin_BC(u, N, dx, alpha, beta, q, x_int, a=1, b_coeff=1, c=0):
    F = np.zeros(N-1)
    q_values = q(x_int) 

    # Dirichlet condition at the left boundary
    F[0] = (u[1] - 2*u[0] + alpha)/dx**2 + q_values[0]

    # Interior points
    for i in range(1, N-2):
        F[i] = (u[i+1] - 2*u[i] + u[i-1])/dx**2  + q_values[i]
    
    # Robin condition at the right boundary
    # Using a backward difference to approximate the derivative at the boundary: (u[N-1] - u[N-2]) / dx
    # The Robin condition: a * u[N-1] + b * (u[N-1] - u[N-2]) / dx = c
    # Rearrange to include in F[N-2]
    F[N-2] = (b_coeff/dx * u[N-2] - (2*b_coeff/dx + a) * u[N-2] + u[N-3])/dx**2 + q_values[N-2] + b_coeff/dx * c/a

    return F


u_guess = x_int

sol = root(Dirichlet_BC, u_guess, args=(N,dx,alpha,beta,q,x_int))
u_int = sol.x
u = np.concatenate(([alpha],u_int,[beta]))
plt.subplot(3, 1, 1)
plt.plot(x, u, '-', label='Numerical Solution')
plt.title('Dirichlet BC')

sol = root(Neumann_BC, u_guess, args=(N,dx,alpha,beta,q,x_int))
u_int = sol.x
u = np.concatenate(([alpha],u_int))
plt.subplot(3, 1, 2)
plt.plot(x[:-1], u, '-', label='Numerical Solution')
plt.title('Neumann BC')

sol = root(Robin_BC, u_guess, args=(N,dx,alpha,beta,q,x_int))
u_int = sol.x
u = np.concatenate(([alpha],u_int))
plt.subplot(3, 1, 3)
plt.plot(x[:-1], u, '-', label='Numerical Solution')
plt.title('Robin BC')

plt.show()
# Dirichlet, Neumann, and Robin boundary conditions for an ODE with a linear source term done
# Next: Nonlinear source term q(u) and a nonlinear ODE

def q_nonlinear(x,u):
    return np.exp(2*u) 

def Dirichlet_BC_nonlinear(u, N, dx, alpha, beta, q, x_int):
    F = np.zeros(N-1)
    q_values = q(x_int,u) 

    F[0] = (u[1] - 2*u[0] + alpha)/dx**2 + q_values[0]

    for i in range(1,N-2):
        F[i] = (u[i+1] - 2*u[i] + u[i-1])/dx**2  + q_values[i]
    
    F[N-2] = (beta - 2*u[N-2] + u[N-3])/dx**2  + q_values[N-2]

    return F

def Neumann_BC_nonlinear(u, N, dx, alpha, beta, q, x_int):
    F = np.zeros(N-1)
    q_values = q(x_int,u) 

    # Dirichlet condition at the left boundary
    F[0] = (u[1] - 2*u[0] + alpha)/dx**2 + q_values[0]

    # Interior points
    for i in range(1, N-2):
        F[i] = (u[i+1] - 2*u[i] + u[i-1])/dx**2  + q_values[i]
    
    # Neumann condition at the right boundary
    # Approximating u'(b) = beta with a backward difference: (u[N-1] - u[N-2]) / dx = beta
    # Rearrange to express u[N-1] in terms of u[N-2] and beta, then substitute in the discretized equation
    F[N-2] = (u[N-2] - 2*u[N-2] + u[N-3])/dx**2 + 2*beta/dx + q_values[N-2]

    return F

def Robin_BC_nonlinear(u, N, dx, alpha, beta, q, x_int, a=2, b_coeff=0.2, c=3):
    F = np.zeros(N-1)
    q_values = q(x_int,u) 

    # Dirichlet condition at the left boundary
    F[0] = (u[1] - 2*u[0] + alpha)/dx**2 + q_values[0]

    # Interior points
    for i in range(1, N-2):
        F[i] = (u[i+1] - 2*u[i] + u[i-1])/dx**2  + q_values[i]
    
    # Robin condition at the right boundary
    # Using a backward difference to approximate the derivative at the boundary: (u[N-1] - u[N-2]) / dx
    # The Robin condition: a * u[N-1] + b * (u[N-1] - u[N-2]) / dx = c
    # Rearrange to include in F[N-2]
    F[N-2] = (b_coeff/dx * u[N-2] - (2*b_coeff/dx + a) * u[N-2] + u[N-3])/dx**2 + q_values[N-2] + b_coeff/dx * c/a

    return F

u_guess = x_int

sol = root(Dirichlet_BC_nonlinear, u_guess, args=(N,dx,alpha,beta,q_nonlinear,x_int))
u_int = sol.x
u = np.concatenate(([alpha],u_int,[beta]))
plt.subplot(3, 1, 1)
plt.plot(x, u, '-', label='Numerical Solution')
plt.title('Dirichlet BC')

sol = root(Neumann_BC_nonlinear, u_guess, args=(N,dx,alpha,beta,q_nonlinear,x_int))
u_int = sol.x
u = np.concatenate(([alpha],u_int))
plt.subplot(3, 1, 2)
plt.plot(x[:-1], u, '-', label='Numerical Solution')
plt.title('Neumann BC')

sol = root(Robin_BC_nonlinear, u_guess, args=(N,dx,alpha,beta,q_nonlinear,x_int))
u_int = sol.x
u = np.concatenate(([alpha],u_int))
plt.subplot(3, 1, 3)
plt.plot(x[:-1], u, '-', label='Numerical Solution')
plt.title('Robin BC')

plt.show()



#Robin isnt working, its got the same output as the dirichlet




