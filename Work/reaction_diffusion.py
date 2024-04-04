import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class BoundaryCondition:
    def __init__(self, bc, value):
        self.bc = bc
        self.value = value

class DiffusionProblem:
    def __init__(self, N, a, b, D, q, initial_condition, boundary_conditions, time_span, t_eval, method='explicit_euler'):
        self.N = N
        self.a = a
        self.b = b
        self.D = D
        self.q = q
        self.initial_condition = initial_condition
        self.boundary_conditions = boundary_conditions
        self.time_span = time_span
        self.t_eval = t_eval
        self.method = method

def diffusion_rhs(t, U, D, q, a, b, N, left_BC, right_BC):
    dUdt = np.zeros_like(U)

    x = np.linspace(a, b, N+1)
    dx = x[1] - x[0]

    # Interior points
    for i in range(1, N):
        dUdt[i] = D * (U[i+1] - 2*U[i] + U[i-1]) / dx**2 + q(0, x[i], U[i], a, b)

    # Apply boundary conditions
    if left_BC.bc == 'dirichlet':
        dUdt[0] = left_BC.value
    elif left_BC.bc == 'neumann':
        #dUdt[0] = left_BC.value
        dUdt[0] = (U[1] - U[0]) / dx 
    elif left_BC.bc == 'robin':
        dUdt[0] = (U[1] - U[0]) / dx + left_BC.value * (U[0] - a)

    if right_BC.bc == 'dirichlet':
        #dUdt[N] = (U[N] - U[N-1]) / dx
        dUdt[N] = 0
    elif right_BC.bc == 'neumann':
        dUdt[N] = right_BC.value
    elif right_BC.bc == 'robin':
        dUdt[N] = (U[N] - U[N-1]) / dx + right_BC.value * (b - U[N])

    return dUdt

def explicit_euler_step(U, D, q, dt, a, b, N, left_BC, right_BC):
    U_new = np.zeros_like(U)

    # Apply boundary conditions
    if left_BC.bc == 'dirichlet':
        U_new[0] = left_BC.value
    elif left_BC.bc == 'neumann':
        U_new[0] = U[0] + dt * left_BC.value
    elif left_BC.bc == 'robin':
        U_new[0] = U[0] + dt * (left_BC.value(0) * (U[0] - a))

    dx = (b - a) / N
    x = np.linspace(a, b, N+1)

    for i in range(1, N):
        U_new[i] = U[i] + dt * D * (U[i+1] - 2*U[i] + U[i-1]) / dx**2 + dt * q(0, x[i], U[i], a, b)

    if right_BC.bc == 'dirichlet':
        U_new[N] = right_BC.value
    elif right_BC.bc == 'neumann':
        U_new[N] = U[N] + dt * right_BC.value
    elif right_BC.bc == 'robin':
        U_new[N] = U[N] + dt * (right_BC.value(0) * (b - U[N]))

    return U_new


def solve_diffusion(problem):
    a = problem.a
    N = problem.N
    b = problem.b
    D = problem.D
    q = problem.q
    initial_condition = problem.initial_condition
    boundary_conditions = problem.boundary_conditions
    time_span = problem.time_span
    t_eval = problem.t_eval
    method = problem.method
    print(method)

    dx = (b - a) / N
    x = np.linspace(a, b, N+1)

    if method == 'explicit_euler':
        dt = time_span[1] / (len(t_eval) - 1)
        U = np.zeros((len(t_eval), N+1))
        U[0] = initial_condition(x)

        for i in range(len(t_eval) - 1):
            U[i+1] = explicit_euler_step(U[i], D, q, dt, a, b, N, boundary_conditions[0], boundary_conditions[1])

    elif method == 'solve_ivp':
        def diffusion_system(t, U):
            return diffusion_rhs(t, U, D, q, a, b, N, boundary_conditions[0], boundary_conditions[1])

        U = solve_ivp(diffusion_system, time_span, initial_condition(x), t_eval=t_eval, method='RK45').y
        U = U.T

    else:
        raise ValueError("Invalid method. Choose 'explicit_euler' or 'solve_ivp'.")

    return x, t_eval, U



def animate_solution(x, t_eval, U):
    fig, ax = plt.subplots()
    line, = ax.plot(x, U[0], color='blue')

    def update(frame):
        line.set_ydata(U[frame])
        line.set_xdata(x)  
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(t_eval), blit=True)
    plt.show()

def plot_solution(x, t_eval, U):
    plt.figure(figsize=(10, 6))
    for i in range(len(t_eval)):
        plt.plot(x, U[i], label=f't={t_eval[i]:.2f}')
    plt.xlabel('x')
    plt.ylabel('U')
    plt.legend()
    plt.title('Diffusion Solution')
    plt.show()

def main():
    a = 0
    b = 1
    D = 1
    N = 20
    q = lambda t, x, U, a, b: 0#U * np.sin(x)

    def initial_condition(x):
        return np.sin(np.pi * x)

    def left_boundary_condition(t):
        return 0

    def right_boundary_condition(t):
        return 1

    time_span = (0, 3)
    dt = 0.00125
    #value of dt where explicit euler method is stable critera:
    #dt <= dx**2/(2*D)
    dt_max = ((b-a)/N)**2/ (2 * D)
    print(dt_max)

    t_eval = np.arange(*time_span, dt)

    boundary_conditions = (BoundaryCondition('dirichlet', value=left_boundary_condition(0)),
                           BoundaryCondition('neumann', value=right_boundary_condition(0)))

    problem = DiffusionProblem(N,a, b, D, q, initial_condition, boundary_conditions, time_span, t_eval)

    x, t_eval, U = solve_diffusion(problem)
    
    animate_solution(x, t_eval, U)
    plt.show()

if __name__ == '__main__':
    main()
