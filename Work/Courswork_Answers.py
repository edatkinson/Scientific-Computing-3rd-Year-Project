
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
from ode_solver import solve_ode

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
from bvp_and_shooting import phase_condition, shoot, limit_cycle_finder, phase_portrait_plotter, orbit


pars = [3] #A & B
y0 = np.array([2,3,7])
sol = limit_cycle_finder(brusselator, y0,phase_condition,pars,test=False)
cycle = orbit(brusselator, sol[:-1], sol[-1], pars)
print(f'The starting point: {", ".join([f"{val:.2f}" for val in sol[:-1]])} values and period: {sol[-1]:.2f} for the Brusselator orbit')
fig1 = phase_portrait_plotter(cycle) #plot the limit cycle
plt.show()


# %%
# (c) For 2 ≤ B ≤ 3, use natural-parameter continuation to find the branch of limit cycles
# that emerge from the Hopf bifurcation at B = 2.
from numerical_continuation import numerical_continuation, natural_plotter

x0 = np.array([0.37, 3.5, 7.15])    
par_array = [3]  # Start parameter value
par_index = 0

par_nat, sol_nat = numerical_continuation(brusselator, 
    'natural', 
    x0, 
    par_array, 
    par_index, 
    [3, 1.5], #Bounds
    [200, 30], #Max steps: [PAL, Natural]
    shoot, #discretization
    fsolve, #solver
    phase_condition=phase_condition, 
    increase=False) #increase parameter


# #Bifurcation diagram of Limit cycle and equilibria

natural_plotter(par_nat, sol_nat)
#add titles n shiz



# %%
# Pseudo Arc Length - verify that natural continuation is working and PAL is working (It is)
from numerical_continuation import pseudo_plotter


par_pseudo, sol_pseudo = numerical_continuation(brusselator, 
    'pseudo', 
    x0, 
    par_array, 
    par_index, 
    [3, 1.5],
    [200, 30], 
    shoot, 
    fsolve, 
    phase_condition=phase_condition, 
    increase=False)

pseudo_plotter(par_pseudo, sol_pseudo) #using solve_ivp instead of solve_ode somehow makes PAL miss the limit cycle

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

from bvp_and_shooting import phase_condition, shoot, limit_cycle_finder, phase_portrait_plotter, orbit

pars = [1] # beta
y0 = np.array([1, 0, -1, 10])
sol = limit_cycle_finder(hopf_bifurcation_3d, y0,phase_condition,pars,test=False)
cycle = orbit(hopf_bifurcation_3d, sol[:-1], sol[-1], pars)
print(f'The starting point: {", ".join([f"{val:.2f}" for val in sol[:-1]])} values and period: {sol[-1]:.2f} for the Hopf Bifurcation 3D orbit')
fig1 = phase_portrait_plotter(cycle) #plot the limit cycle
plt.show()
# Need to verify that this is the limit cycle found in a)
# How to do this?

# %%
# (c) For β ≤ 1, from the starting point found in (b) use pseudo-arclength continuation to
# find the branch of limit cycles that emerge from the Hopf bifurcation at β

x0 = sol #starting point from b) 
par_array = [2]  # Start parameter value - becomes redundant anyway as we start from max unless there are more parameters
par_index = 0

par_pseudo, sol_pseudo = numerical_continuation(hopf_bifurcation_3d, 
    'pseudo', 
    x0, 
    par_array, 
    par_index, 
    [1, -0.6], #Bounds affect the step size
    [200, 30], #Max steps: [PAL, Natural]. Also affects the step size  
    shoot, 
    fsolve, 
    phase_condition=phase_condition, 
    increase=False)

pseudo_plotter(par_pseudo, sol_pseudo)
#Solve_IVP is better than solve_ode in this case due to squared terms in the equations



# %%
#Question 3:

