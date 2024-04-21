
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

#Robin isnt working, its got the same output as the dirichlet
# Verfiy my methods are working by using bvp solvers and comparing the results
import fipy as fp


# Constants
L = 6
D = 0.01
T = 100
nx = 100  # number of spatial points
dx = L / nx  # spatial resolution

# Create mesh and variables
mesh = fp.Grid1D(dx=dx, nx=nx)
u = fp.CellVariable(name="u", mesh=mesh, value=0.0)

# PDE definition
eq = (fp.TransientTerm() == D * fp.DiffusionTerm(coeff=1) + (1 - u)**2 * fp.CellVariable(mesh=mesh, value=np.exp(-mesh.cellCenters[0])))

# Boundary conditions
u.faceGrad.constrain([0], where=mesh.facesLeft)
u.faceGrad.constrain([0], where=mesh.facesRight)

# Solve
timeStepDuration = T / 1000
steps = int(T / timeStepDuration)
for step in range(steps):
    eq.solve(var=u, dt=timeStepDuration)

x = np.linspace(0, L, nx)
u_values = u.value
plt.plot(x, u_values, label='u(x)')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution of the PDE at final time step')
plt.legend()
plt.grid(True)
plt.show()

u_L_T = u_values[-1]  # Value of u at x=L, T=T
print("The value of u at (L, T) is:", u_L_T)



# Given constants
D = 0.5
L = 2
T = 0.5
Nx = 101  # Number of spatial points
dx = L / (Nx - 1)  # Spatial step size
dt = T / 100  # Time step size, 100 time steps

# Create mesh
mesh = fp.Grid1D(dx=dx, nx=Nx)

# Create variable
u = fp.CellVariable(name="u", mesh=mesh)

# Initial condition
u.setValue(0.5 * mesh.cellCenters[0] * (L - mesh.cellCenters[0]))

# Boundary conditions
u.faceGrad.constrain(1, where=mesh.facesLeft)  # Neumann boundary condition at x=0
u.constrain(0, where=mesh.facesRight)  # Dirichlet boundary condition at x=L

# Create the PDE
eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=D)

# Time-stepping
time = 0
while time < T:
    time += dt
    eq.solve(var=u, dt=dt)

# Check final time to not exceed T due to possible floating-point arithmetic issues
if time != T:
    overshoot = time - T
    eq.solve(var=u, dt=-overshoot)  # Adjust the last time step to be exact

# Plotting
x = np.linspace(0, L, Nx)
u_values = u.value
plt.plot(x, u_values, label='u(x)')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution of the PDE at T = 0.5')
plt.legend()
plt.grid(True)
plt.show()


P = 1  # Peclet number is given as a constant
from scipy.integrate import solve_bvp
# Define the differential equation
def odefun(x, y):
    return np.vstack((y[1], P*y[1] - P))

# Define the boundary conditions
def bc(ya, yb):
    return np.array([ya[0], yb[0] - 0.5])

# Initial mesh and initial guess for y
x = np.linspace(0, 1, 5)
y_guess = np.zeros((2, x.size))

# Solve the BVP without additional parameters
sol = solve_bvp(odefun, bc, x, y_guess)

# Create a finer mesh for plotting
x_plot = np.linspace(0, 1, 100)
y_plot = sol.sol(x_plot)[0]

# Plot the solution
plt.plot(x_plot, y_plot, label='u(x)')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution of the steady reaction-convection-diffusion equation')
plt.legend()
plt.show()
