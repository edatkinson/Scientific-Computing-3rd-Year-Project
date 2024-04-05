import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class BoundaryCondition:
    def __init__(self, bc, value):
        self.bc = bc
        self.value = value

class DiffusionProblem:
    def __init__(self, N, a, b, D, q, initial_condition, boundary_conditions, time_span, t_eval, method):
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
        U[0] = left_BC.value
    elif left_BC.bc == 'neumann':
        dUdt[0] = left_BC.value
        #dUdt[0] = (U[1] - U[0]) / dx  
        #dUdt[0] = D * (U[1] - U[0]) / dx**2 + left_BC.value
    elif left_BC.bc == 'robin':
        dUdt[0] = (U[1] - U[0]) / dx + left_BC.value * (U[0] - a)

    if right_BC.bc == 'dirichlet':
        U[N] = right_BC.value

    elif right_BC.bc == 'neumann':
        dUdt[N] = right_BC.value 
        #dUdt[N] = (U[N] - U[N-1]) / dx 
        #dUdt[N] = D * (U[N] - U[N-1]) / dx**2 + right_BC.value
    elif right_BC.bc == 'robin':
        dUdt[N] = (U[N] - U[N-1]) / dx + right_BC.value * (b - U[N])

    return dUdt

def explicit_euler_step(U, D, q, dt, a, b, N, left_BC, right_BC):
    U_new = np.zeros_like(U)

    dx = (b - a) / N
    x = np.linspace(a, b, N+1)

    # Apply boundary conditions
    if left_BC.bc == 'dirichlet':
        #U_new[0] = left_BC.value
        U[0] = left_BC.value

    for i in range(1, N):
        U_new[i] = U[i] + dt * D * (U[i+1] - 2*U[i] + U[i-1]) / dx**2 + dt * q(0, x[i], U[i], a, b)
    
    if left_BC.bc == 'neumann':
        #U_new[0] = U[0] + dt*D/dx**2 * (U[1] -2* U[0] + left_BC.value)
        #U_new[0] = U[0] + dt * D / dx * (left_BC.value)
        U_new[0] = U[0] + dt * D / dx**2 * (U[1] - U[0]) + dt * left_BC.value / dx
    elif left_BC.bc == 'robin':
        U_new[0] = U[0] + dt * (left_BC.value(0) * (U[0] - a))
    
    if right_BC.bc == 'dirichlet':
        U_new[N] = right_BC.value
    elif right_BC.bc == 'neumann':
        #U_new[N] = U[N] + D*dt / dx * right_BC.value
        #U_new[N] = U[N] + dt*D/dx**2 * (right_BC.value - 2*U[N] + U[N-1])
        U_new[N] = U[N] + dt * D / dx**2 * (U[N-1] - U[N]) + dt * right_BC.value / dx
    elif right_BC.bc == 'robin':
        U_new[N] = U[N] + dt * (right_BC.value(0) * (b - U[N]))

    return U_new

def diffusion_implicit_euler(D, q, bc_left, bc_right, u0, dt, dx, T, a, b):
    Nt = int(T/dt) + 1  # Number of time steps
    Nu = int((b - a)/dx) + 1  # Number of spatial points
    u = np.zeros((Nt, Nu))
    x = np.linspace(a, b, Nu)  # Spatial grid
    u[0] = u0(x)  # Applying the initial condition

    # Adjusted matrix A for Implicit Euler
    r = D * dt / dx**2
    A = np.diag((1 + 2*r) * np.ones(Nu)) - np.diag(r * np.ones(Nu - 1), -1) - np.diag(r * np.ones(Nu - 1), 1)

    # Apply Dirichlet boundary conditions in A
    A[0, 0], A[-1, -1] = 1, 1
    A[0, 1], A[-1, -2] = 0, 0  # Ensuring no flux contribution from outside the domain

    for n in range(1, Nt):
        un = u[n-1]
        b = un   # RHS based on previous timestep, scaled by 1/dt

        # Add the contribution from the q term if it's not trivial
        b += q(n*dt, x, un, a, b) * dt  # q term scaled by dt for the implicit method

        # Apply boundary conditions
        if bc_left.bc == 'dirichlet':
            b[0] = bc_left.value
        elif bc_left.bc == 'neumann':
            A[0, 0] = -1 / dx
            A[0, 1] = 1 / dx
            b[0] += bc_left.value * dx / D  # Adjust for Neumann on the left

        if bc_right.bc == 'dirichlet':
            b[-1] = bc_right.value
        elif bc_right.bc == 'neumann':
            b[-1] += bc_right.value * dx / D  # Adjust for Neumann on the right

        # Solve the system for this timestep
        u[n] = np.linalg.solve(A, b)

    return u


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
        t_eval = np.linspace(*time_span, N)
        def diffusion_system(t, U):
            return diffusion_rhs(t, U, D, q, a, b, N, boundary_conditions[0], boundary_conditions[1])

        U = solve_ivp(diffusion_system, time_span, initial_condition(x), t_eval=t_eval, method='RK45').y
        U = U.T
    
    elif method == 'implicit_euler':
        dt = 0.01
        U = diffusion_implicit_euler(D, q, boundary_conditions[0], boundary_conditions[1], initial_condition, dt, dx, time_span[1], a, b)

    else:
        raise ValueError("Invalid method. Choose 'explicit_euler' or 'solve_ivp'.")

    return x, t_eval, U




def animate_solution(x, t_eval, U,title):
    fig, ax = plt.subplots()
    line, = ax.plot(x, U[0], color='blue')
    plt.xlim(-0.5,2.5)
    plt.ylim(-0.5,1.5)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('U(x,t)')
    plt.grid(True)
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
    T = 2

    q = lambda t, x, U, a, b: 0*x #U * np.sin(x)

    def initial_condition(x):
        return np.sin(np.pi *(x-a)/(b-a))

    def left_boundary_condition(t):
        return 0

    def right_boundary_condition(t):
        return 1
    dx = (b - a) / N
    time_span = (0, T)
    dt = 0.00125
    #value of dt where explicit euler method is stable critera:
    #dt <= dx**2/(2*D)
    dt_max = ((b-a)/N)**2/ (2 * D)
    print(f'Using the Criterion to maintian stability for the Explicit Euler Method, the Maxium value of Delta T = {dt_max}')
    dt = 0.5*dt_max

    t_eval = np.arange(*time_span, dt)

    boundary_conditions = (BoundaryCondition('dirichlet', value=left_boundary_condition(0)),
                           BoundaryCondition('dirichlet', value=right_boundary_condition(0)))

    #IVP Method
    problem_ivp = DiffusionProblem(N,a, b, D, q, initial_condition, boundary_conditions, time_span, t_eval, method='solve_ivp')
    #Explicit Euler Method
    problem_euler = DiffusionProblem(N,a, b, D, q, initial_condition, boundary_conditions, time_span, t_eval, method='explicit_euler')
    #Implicit Euler Method
    x,t,U = solve_diffusion(DiffusionProblem(N,a, b, D, q, initial_condition, boundary_conditions, time_span, t_eval, method='implicit_euler'))
    
    x = np.linspace(a, b, N+1)
    x_ivp, t_eval_ivp, U_ivp = solve_diffusion(problem_ivp)
    x_euler, t_eval_euler, U_euler = solve_diffusion(problem_euler)

    animate_solution(x_ivp, t_eval_ivp, U_ivp, "IVP Method")
    
    animate_solution(x_euler, t_eval_euler, U_euler, "Explicit Euler Method")

    

    animate_solution(x, t, U, 'Implicit Euler Method')


if __name__ == '__main__':
    main()


# Simulates Exercise 1 analytical sol from week 20

# # Parameters
# D = 1  # Diffusion coefficient
# a = 0   # Left boundary
# b = 1   # Right boundary
# x = np.linspace(a, b, 100)  # Space grid
# t_start, t_end, t_step = 0, 2, 0.01  # Time range and step

# # Pre-calculate exact solution for all time steps
# u_exact = np.zeros((int((t_end - t_start) / t_step) + 1, x.size))
# for i, t in enumerate(np.arange(t_start, t_end + t_step, t_step)):
#     u_exact[i, :] = np.exp(-D * np.pi ** 2 * t / ((b - a) ** 2)) \
#                    * np.sin(np.pi * (x - a) / (b - a))

# # Animation function
# def update(frame):
#     line.set_ydata(u_exact[frame, :])
#     return line,

# # Plot setup
# fig, ax = plt.subplots()
# line, = ax.plot(x, u_exact[0, :])
# ax.set_xlabel('x')
# ax.set_ylabel('u(x, t)')
# ax.set_xlim(a, b)
# ax.set_ylim(-1.2, 1.2)

# # Animation creation
# ani = FuncAnimation(fig, update, frames=int((t_end - t_start) / t_step) + 1, interval=100, blit=True)

# # Display the animation
# plt.show()
