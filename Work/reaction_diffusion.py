import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import root
from scipy.linalg import solve_banded
from matplotlib.animation import FuncAnimation
import time


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
        dUdt[i] = D * (U[i+1] - 2*U[i] + U[i-1]) / dx**2 + q(t, x[i], U[i])

    # Apply boundary conditions
    if left_BC.bc == 'dirichlet':
        U[0] = left_BC.value
        
    elif left_BC.bc == 'neumann':

        dUdt[0] = D * (U[1] - U[0] + dx * left_BC.value) / dx**2
        #U[0] = U[1] - left_BC.value * dx


    elif left_BC.bc == 'robin':
        dUdt[0] = (U[1] - U[0]) / dx + left_BC.value * (U[0] - a)

    if right_BC.bc == 'dirichlet':
        U[N] = right_BC.value

    elif right_BC.bc == 'neumann':
        dUdt[-1] = D * (right_BC.value * dx - U[-1] + U[-2]) / dx**2
        #U[N] = U[N-1] + right_BC.value * dx

    elif right_BC.bc == 'robin':
        dUdt[N] = (U[N] - U[N-1]) / dx + right_BC.value * (b - U[N])

    return dUdt


def explicit_euler_step(U, D, q, dt, a, b, N, left_BC, right_BC, T):
    dx = (b - a) / N
    U_new = U.copy()

    # Update interior points
    for i in range(1, N):
        U_new[i] = U[i] + dt * (D * (U[i+1] - 2*U[i] + U[i-1]) / dx**2 + q(T, (i*dx) + a, U[i]))

    # Apply Neumann BCs on the boundaries
    if left_BC.bc == 'neumann':
        U_new[0] = U_new[1] + left_BC.value * dx
    if right_BC.bc == 'neumann':
        U_new[-1] = U_new[-2] - right_BC.value * dx

    # Apply Dirichlet BCs on the boundaries
    if left_BC.bc == 'dirichlet':
        U_new[0] = left_BC.value
    if right_BC.bc == 'dirichlet':
        U_new[-1] = right_BC.value

    return U_new


def linalg_implicit(D, f, u0, bc_left, bc_right, a, b, dx, dt, T):
    """
    Solves a non-linear heat equation u_t = D*u_xx + f(t, x, u) using the implicit Euler method,
    with support for Neumann boundary conditions.

    Parameters:
    - D: Diffusion coefficient.
    - f: Non-linear source term function, f(t, x, u).
    - u0: Initial condition function, u0(x).
    - bc_left, bc_right: Boundary condition objects for left and right boundaries.
      Each BC object should have 'type' ('dirichlet' or 'neumann') and 'value'.
    - a, b: Spatial domain boundaries.
    - dx: Spatial step size.
    - dt: Time step size.
    - T: Final time.

    Returns:
    - u: Solution matrix, where each row represents the solution at a time step.
    - x: Spatial grid points.
    """
    # Define the spatial and temporal grids
    x = np.arange(a, b + dx, dx)
    t = np.arange(0, T + dt, dt)
    Nx = len(x)
    Nt = len(t)
    r = D * dt / dx**2

    # Initialize the solution matrix
    u = np.zeros((Nt, Nx))
    u[0, :] = u0(x)

    # Set up the coefficient matrix for the implicit scheme
    A = np.zeros((Nx, Nx))
    np.fill_diagonal(A, 1 + 2*r)
    np.fill_diagonal(A[1:, :], -r)
    np.fill_diagonal(A[:, 1:], -r)

    # Adjust the matrix for boundary conditions
    if bc_left.bc == 'neumann':
        A[0, 0] = 1 + r  # Adjust the coefficient for Neumann BC
    else:  # Dirichlet
        A[0, :] = 0
        A[0, 0] = 1

    if bc_right.bc == 'neumann':
        A[-1, -1] = 1 + r  # Adjust the coefficient for Neumann BC
    else:  # Dirichlet
        A[-1, :] = 0
        A[-1, -1] = 1

    for n in range(1, Nt):
        # Evaluate the non-linear term including dependency on x and t
        non_linear_term = f(t[n], x, u[n-1, :])
        
        # Construct the right-hand side F for the linear system
        F = u[n-1, :] + dt * non_linear_term

        # Apply boundary conditions
        if bc_left.bc == 'neumann':
            F[0] += r * bc_left.value * dx  # Adjust F for Neumann BC
        else:  # Dirichlet
            F[0] = bc_left.value 
        

        if bc_right.bc == 'neumann':
            F[-1] += r * bc_right.value * dx  # Adjust F for Neumann BC
        else:  # Dirichlet
            F[-1] = bc_right.value

        # Solve the system A*u[n, :] = F
        u[n, :] = np.linalg.solve(A, F)

    return u, x, t


def diffusion_implicit_euler(D, q, bc_left, bc_right, u0, dt, dx, T, a, b):
    Nt = int(T/dt) + 1  # Number of time steps
    Nu = int((b - a)/dx) + 1  # Number of spatial points
    u = np.zeros((Nt, Nu))
    x = np.linspace(a, b, Nu)  # Spatial grid
    u[0, :] = u0(x)  # Applying the initial condition

    # Coefficient for diffusion term
    r = D * dt / dx**2

    # Define the function for fsolve
    def solve_u_next(U_next, U_prev, n):
        non_linear_term = q(n*dt, x, U_next)
        # Construct the diffusion part using central difference (Laplacian)
        diffusion_term = np.zeros_like(U_next)
        diffusion_term[1:-1] = D * (U_next[:-2] - 2*U_next[1:-1] + U_next[2:]) / dx**2
        
        # Combine the terms and subtract from previous U to form residual for fsolve
        F = U_next - U_prev - dt * (diffusion_term + non_linear_term)

        # Apply Neumann BC correctly for the left boundary
        if bc_left.bc == 'dirichlet':
            F[0] = U_next[0] - bc_left.value
        elif bc_left.bc == 'neumann':
            # Use a backward difference for left boundary to maintain consistency
            F[0] = (U_next[0] - U_next[1]) / dx - bc_left.value  

        # Apply Neumann BC correctly for the right boundary
        if bc_right.bc == 'dirichlet':
            F[-1] = U_next[-1] - bc_right.value
        elif bc_right.bc == 'neumann':
            # Use a forward difference for right boundary to maintain consistency
            F[-1] = (U_next[-1] - U_next[-2]) / dx - bc_right.value
        
        return F

    for n in range(1, Nt):
        U_prev = u[n-1, :]
        # Use fsolve to find U_next
        U_next_initial_guess = U_prev  # Initial guess for U_next
        # Solve for U_next using fsolve
        #u[n, :], infodict, ier, mesg = fsolve(solve_u_next, U_next_initial_guess, args=(U_prev, n), full_output=True)
        sol = root(solve_u_next, U_prev, args=(U_prev, n))
        if not sol.success:
            print(f"Root did not converge at time step {n}, message: {sol.message}")
            break
        u[n, :] = sol.x 

    return u, x


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
    

    dx = (b - a) / N
    x = np.linspace(a, b, N+1)

    if method == 'explicit_euler':
        dt = time_span[1] / (len(t_eval) - 1)
        N = len(initial_condition(x)) - 1  # Assuming initial_condition(x) returns an array of length N+1
        U = np.zeros((len(t_eval), N+1))
        U[0] = initial_condition(x)

        for i in range(len(t_eval) - 1):
            U[i+1] = explicit_euler_step(U[i], D, q, dt, a, b, N, boundary_conditions[0], boundary_conditions[1], t_eval[i])
        print(method)
    elif method == 'solve_ivp':
        t_eval = np.linspace(*time_span, N)
        def diffusion_system(t, U):
            return diffusion_rhs(t, U, D, q, a, b, N, boundary_conditions[0], boundary_conditions[1])

        U = solve_ivp(diffusion_system, time_span, initial_condition(x), t_eval=t_eval, method='RK45').y
        U = U.T
        print(method)
    elif method == 'implicit_euler':
        dt = time_span[1] / (len(t_eval) - 1) 
        U,x = diffusion_implicit_euler(D, q, boundary_conditions[0], boundary_conditions[1], initial_condition, dt, dx, time_span[1], a, b)
        print(method)
    else:
        raise ValueError("Invalid method. Choose 'explicit_euler' , 'solve_ivp' or implicit_euler.")
    
    return x, t_eval, U




def animate_solution(x, t_eval, U,title):
    fig, ax = plt.subplots()
    t_eval = np.linspace(t_eval[0], t_eval[-1], len(U))
    line, = ax.plot(x, U[0], color='blue')
    ax.set_xlim(np.min(x)-0.5, np.max(x)+0.5)
    ax.set_ylim(np.min(U), np.max(U)+0.5)
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


from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection

def plot_multiple_solutions_3d(solution_sets, titles):
    """
    Plots multiple 3D solutions for comparison.

    :param solution_sets: A list of tuples, each containing (x, t_eval, U) for a solution.
    :param titles: A list of titles for each subplot.
    """
    n = len(solution_sets)  # Number of solutions to plot
    cols = 2  # You can adjust this based on how you want to display
    rows = (n + 1) // cols  # Calculate rows needed

    fig = plt.figure(figsize=(12, 6))

    for i, ((x, t_eval, U), title) in enumerate(zip(solution_sets, titles)):
        #ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        
        # Ensure t_eval is the correct length
        t_eval = np.linspace(t_eval[0], t_eval[-1], len(U))

        # Generate meshgrid
        T, X = np.meshgrid(t_eval, x)  # Note the order to match U's shape

        # Ensure U is transposed if necessary to match the shapes of X and T
        U_correct_shape = U.T if U.shape[0] == len(x) else U
        if U_correct_shape.shape != X.shape:
            U_correct_shape = U_correct_shape.T 

        # Plot the surface
        surf = ax.plot_surface(X, T, U_correct_shape, cmap='viridis', edgecolor='none')

        # Setting labels
        ax.set_xlabel('Space domain (x)')
        ax.set_ylabel('Time domain (t)')
        ax.set_zlabel('U(x,t)')
        ax.set_title(title)

    fig.colorbar(surf, ax=fig.axes, shrink=0.5, aspect=5, location='bottom')
    #plt.tight_layout()
    plt.show()


def main():
    a = 0
    b = 2
    D = 1
    N = 20
    T = 0.5

    q = lambda t, x, U: x*0

    def initial_condition(x):
        return np.sin(np.pi * (x) / (2))

    def left_boundary_condition(t):
        return 0

    def right_boundary_condition(t):
        return 0
    dx = (b - a) / N
    time_span = (0, T)
    dt = 0.00125
    #value of dt where explicit euler method is stable critera:
    #dt <= dx**2/(2*D)
    dt_max = ((b-a)/N)**2/ (2 * D)
    print(f'Using the Criterion to maintian stability for the Explicit Euler Method, the Maxium value of Delta t = {dt_max}') 
    dt = 0.5*dt_max

    t_eval = np.arange(*time_span, dt)

    boundary_conditions = (BoundaryCondition('dirichlet', value=left_boundary_condition(0)),
                           BoundaryCondition('dirichlet', value=left_boundary_condition(0)))

    start_time = time.time()
    problem_ivp = DiffusionProblem(N,a, b, D, q, initial_condition, boundary_conditions, time_span, t_eval, method='solve_ivp')
    x_ivp, t_eval_ivp, U_ivp = solve_diffusion(problem_ivp)
    ivp_time = time.time() - start_time

    #Explicit Euler Method
    start_time = time.time()
    problem_euler = DiffusionProblem(N,a, b, D, q, initial_condition, boundary_conditions, time_span, t_eval, method='explicit_euler')
    x_euler, t_eval_euler, U_euler = solve_diffusion(problem_euler)
    euler_time = time.time() - start_time

    #Implicit Euler Method using Fsolve - when q is exponential in the non-linear term fsolve does not converge well.
    start_time = time.time()
    x_imp,t_imp,U_imp = solve_diffusion(DiffusionProblem(N,a, b, D, q, initial_condition, boundary_conditions, time_span, t_eval, method='implicit_euler'))
    implicit_time = time.time() - start_time

    #Implicit Euler Method using linalg.solve
    start_time = time.time()
    U,x,t = linalg_implicit(D, q, initial_condition, boundary_conditions[0], boundary_conditions[1], a, b, dx, dt, T)
    implicit_time_linalg = time.time() - start_time



    print(f"IVP Method execution time: {ivp_time:.4f} seconds")
    print(f"Explicit Euler Method execution time: {euler_time:.4f} seconds")
    print(f"Implicit Euler Method execution time: {implicit_time:.4f} seconds")
    print(f"Implicit Linalg.solve execution time: {implicit_time_linalg:.4f} seconds")


    solution_sets = [(x_ivp, t_eval_ivp, U_ivp), (x_euler, t_eval_euler, U_euler), (x, t, U), (x_imp,t_imp,U_imp)]
    titles = ['IVP Method', 'Explicit Euler Method', 'Linalg Implicit Euler Method ','Root Implicit Euler Method']
    plot_multiple_solutions_3d(solution_sets, titles)

    animate_solution(x_ivp, t_eval_ivp, U_ivp, "IVP Method")
    
    animate_solution(x_euler, t_eval_euler, U_euler, "Explicit Euler Method")

    animate_solution(x_imp, t_imp, U_imp, "Root Implicit Euler Method")

    animate_solution(x, t, U, 'Linalg Implicit Euler Method')


if __name__ == '__main__':
    main()

#TODO: Add IMEX ability from week 22 & t dependent BCs
#TODO: Add a Sparse Matrix Solver for the Implicit Euler Method to optimise it
#TODO: changed to root function 

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
