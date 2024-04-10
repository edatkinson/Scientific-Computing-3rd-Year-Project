
# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import scipy.linalg as LA

# %%
#a)
# For B = 3, use a numerical integrator to calculate the solution for 0 ≤ t ≤ 20
# with initial conditions x(0) = 1 and y(0) = 1. Plot the resulting time series as the
# trajectory converges onto a limit cycle, showing both x and y against t.

from Equations_Functions import brusselator
from new_ode_solver import solve_ode

pars = [3] # B (A is fixed = 1)
y0 = np.array([1, 1])
T = 20
t = np.linspace(0, T, 1000)
sol = solve_ode(brusselator, y0, t, "rk4", 0.05, pars)
plt.plot(t, sol[0, :], label='x')
plt.plot(t, sol[1, :], label='y')
plt.xlabel('Time')
plt.ylabel('Brusselator')
plt.legend()
plt.show()

# %%
#b)
# For B = 3, use numerical shooting along with a suitable phase condition to identify
# the coordinates of a starting point along the limit cycle. Determine the oscillation
# period to two decimal places.
from scipy.optimize import fsolve
from bvp_and_shooting import phase_condition, shoot, limit_cycle_finder, phase_portrait_plotter


pars = [3] #A & B
y0 = np.array([2,3,7])
orbit, cycle1 = limit_cycle_finder(brusselator, y0,phase_condition,pars,test=False)
print(f'The starting point: {", ".join([f"{val:.2f}" for val in orbit[:-1]])} values and period: {orbit[-1]:.2f} for the Brusselator orbit')
fig1 = phase_portrait_plotter(cycle1) #plot the limit cycle
plt.show()


# %%
# (c) For 2 ≤ B ≤ 3, use natural-parameter continuation to find the branch of limit cycles
# that emerge from the Hopf bifurcation at B = 2.
from numerical_continuation import natural_continuation, plotter

steps = 70
initial_guess = np.array([0.37, 3.5, 7.15])    #np.array([0.2,0.5,35]) works with mybvp
param_bounds = [3,2] #(max,min)

limit_cycle, param_values, equilibra = natural_continuation(
    brusselator, 
    initial_guess, 
    steps, 
    param_bounds, 
    phase_condition)


# #Bifurcation diagram of Limit cycle and equilibria

plt.figure(figsize=(8, 6))
plt.plot(param_values, limit_cycle[:,0], label='x')
plt.plot(param_values, limit_cycle[:,1], label='y')
plt.xlabel('Parameter', fontsize=12)
plt.ylabel('u', fontsize=12)
plt.title('Natural Continuation Bifurcation Diagram of x and y', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()



# %%
# Pseudo Arc Length - verify that natural continuation is working and PAL is working (It is)
from numerical_continuation import pseudo_continuation
max_steps = 400 #Determines the step size. Needs to cause a small enough step size
x0 = np.array([0.37, 3.5, 7.15])
par_array = [3]  # Start parameter value
vary_par = 0
par_index = 0
min_par, max_par = 2, 3

par_range = [3,1]

par, X_sol = pseudo_continuation(brusselator, 
    x0, 
    par_array,
    par_index, 
    min_par, 
    max_par, 
    max_steps,
    increase=False
)


x,y,X_norm = [], [], []
count = 0
while count < len(X_sol):
    x.append(X_sol[count][0])
    y.append(X_sol[count][1])
    X_norm.append(np.array(LA.norm(X_sol[count][:-1], axis=0, keepdims=True))) #exclding the T value
    count += 1

plt.plot(par, x, label='x')
plt.plot(par, y, label='y')
plt.plot(par, X_norm, label='||x||')
plt.xlabel('Parameter', fontsize=12)
plt.ylabel('X(x,y)', fontsize=12)
plt.title('Pseudo Arc Length Parameter Continuation Bif Diagram', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# %%
#Question 2:
from Equations_Functions import hopf_bifurcation_3d

# (a) For β = 1, use a numerical integrator to calculate the solution for 0 ≤ t ≤ 10 with
# initial conditions x(0) = 1, y(0) = 0, and z(0) = −1. Plot the resulting time series
# as the trajectory converges onto a limit cycle, showing x, y, and z again

pars = [1] # beta
y0 = np.array([1, 0, -1])
T = 10
t = np.linspace(0, T, 1000)
sol = solve_ode(hopf_bifurcation_3d, y0, t, "rk4", 0.01, pars)

plt.plot(t, sol[0, :], label='x')
plt.plot(t, sol[1, :], label='y')
plt.plot(t, sol[2, :], label='z')
plt.xlabel('Time')
plt.ylabel('3D Hopf')
plt.legend()
plt.show()



# %%
# (b) For β = 1, use numerical shooting along with a suitable phase condition to identify
# the coordinates of a starting point along the limit cycle found in (a). Determine the
# oscillation period to two decimal places. Note: there are multiple co-existing limit
# cycles; ensure that you find the one matching the limit cycle in a).

from bvp_and_shooting import phase_condition, shoot, limit_cycle_finder, phase_portrait_plotter

pars = [1] # beta
y0 = np.array([1, 0, -1, 10])
orbit, cycle1 = limit_cycle_finder(hopf_bifurcation_3d, y0,phase_condition,pars,test=False)
print(f'The starting point: {", ".join([f"{val:.2f}" for val in orbit[:-1]])} values and period: {orbit[-1]:.2f} for the Hopf Bifurcation 3D orbit')
fig1 = phase_portrait_plotter(cycle1) #plot the limit cycle
plt.show()
# Need to verify that this is the limit cycle found in a)
# How to do this?

# %%
# (c) For β ≤ 1, from the starting point found in (b) use pseudo-arclength continuation to
# find the branch of limit cycles that emerge from the Hopf bifurcation at β

max_steps = 200 #Depending on the steps - 
x0 = orbit #starting point from b) 
par_array = [2]  # Start parameter value - becomes redundant anyway as we start from max unless there are more parameters
par_index = 0
min_par = -1
max_par = 2


#Better when using solve_ivp instead of solve_ode
par, u_sol = pseudo_continuation(hopf_bifurcation_3d, 
    x0, 
    par_array, 
    par_index, 
    min_par, 
    max_par, 
    max_steps,
    increase=False
)

u1,u2,u3,u_norm = [], [], [], []
count = 0
while count < len(u_sol):
    u1.append(u_sol[count][0])
    u2.append(u_sol[count][1])
    u3.append(u_sol[count][2])
    u_norm.append(np.array(LA.norm(u_sol[count][:-1], axis=0, keepdims=True))) #exclding the T value
    count += 1

plt.plot(par, u1, label='u1')
plt.plot(par, u2, label='u2')
plt.plot(par, u3, label='u3')
plt.plot(par, u_norm, label='norm')
plt.xlabel('Parameter', fontsize=12)
plt.xlim(min_par-2, max_par+2)
plt.ylabel('u', fontsize=12)
plt.title('Pseudo Arc Length Method', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()


from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the parametric plot
ax.plot(par, u1, u2, label='x')
ax.plot(par, u1, u3, label='y')
ax.plot(par, u2, u3, label='z')

# Label the axes
ax.set_xlabel('Parameter')
ax.set_ylabel('X Component')
ax.set_zlabel('Y Component')

# Optionally add a legend
ax.legend()

# Show the plot
plt.show()


# %%
#Question 3:

