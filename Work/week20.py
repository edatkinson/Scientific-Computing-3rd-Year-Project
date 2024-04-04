import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def initial_condition(x, L):
    """
    Defines the initial condition of the system.
    """
    return np.zeros(np.size(x)) #np.sin(np.pi * x / L)


def rhs(t, u, D, dx, N):
    """
    Computes the right-hand side of the ODE system for RK45.
    """
    dudt = np.zeros_like(u)
    dudt[0], dudt[-1] = 0,0 
    for i in range(1, N+1):
        dudt[i] = D * (u[i+1] - 2*u[i] + u[i-1]) / dx**2 + np.exp(2*u[i])
    dudt[0] = dudt[-1] 
    return dudt

def explicit_euler_step(u, D, dx, dt, N):
    """
    Performs one time step using the Explicit Euler method.
    """
    u_new = np.zeros_like(u)
    u[0], u[N+1] = 0,0
    for i in range(1, N+1):
        u_new[i] = u[i] + dt * D * (u[i+1] - 2*u[i] + u[i-1]) / dx**2 + dt*np.exp(2*u[i])
    # Apply Dirichlet boundary conditions
    u_new[0], u_new[N+1] = u[0], u[N+1]
    return u_new

def solve_diffusion(initial_condition,rhs,D, L, N, dx, x, t_span, t_eval,method='explicit_euler'):
    """
    Solves the diffusion equation using the specified method.
    """
    u = initial_condition(x, L)
    if method == 'explicit_euler':
        dt = t_eval[1] - t_eval[0]  # Assuming uniform time steps
        u_all = [u.copy()]
        for _ in t_eval[1:]:
            u = explicit_euler_step(u, D, dx, dt, N)
            u_all.append(u.copy())
        u_all = np.array(u_all).T  # Transpose to match solve_ivp output
    elif method == 'rk45':
        solution = solve_ivp(rhs, t_span, u, args=(D, dx, N), method='RK45', t_eval=t_eval)
        u_all = solution.y
    else:
        raise ValueError("Invalid method. Choose 'explicit_euler' or 'rk45'.")
    return x, t_eval, u_all

def plot_solution(x, t_eval, u_all):
    """
    Plots the solution of the diffusion problem.
    """
    plt.figure(figsize=(10, 6))
    # for i in range(0, len(t_eval), max(1, len(t_eval)//5)):
    #     plt.plot(x, u_all[:, i], label=f't={t_eval[i]:.2f}')
    t_index = np.where(t_eval == 0.5)[0][0]
    plt.plot(x, u_all[:, t_index], label=f't={t_eval[t_index]:.2f}')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.title('Diffusion Solution')
    plt.show()

def main():
    # Parameters
    D = 1.0
    L = 1.0
    N = 100
    dx = L / (N + 1)
    x = np.linspace(0, L, N+2)

    t_span = (0, 1)
    dt_max = dx**2 / (2 * D)

    dt = 0.00004
    if dt <= dt_max:
        print(f"The time step dt = {dt} meets the stability criterion, dt_max = {dt_max:.5f}.")
    else:
        print(f"The time step dt = {dt} does NOT meet the stability criterion, dt_max = {dt_max:.5f}. Adjust dt.")
    
    t_eval = np.arange(t_span[0], t_span[1], dt)
    # Solve and plot using Explicit Euler
    x, t_eval, u_all = solve_diffusion(initial_condition,rhs,D, L, N, dx, x, t_span, t_eval,method='explicit_euler')
    plot_solution(x, t_eval, u_all)

    # Solve and plot using RK45 
    x, t_eval, u_all = solve_diffusion(initial_condition,rhs,D, L, N, dx, x, t_span, t_eval, method='rk45')
    plot_solution(x, t_eval, u_all)

if __name__ == '__main__':
    main()
