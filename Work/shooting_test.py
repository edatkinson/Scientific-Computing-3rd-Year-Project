
# %%
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from bvp_and_shooting import integrate, ode, phase_condition, shoot, limit_cycle_finder, hopf, phase_portrait_plotter

# %%

initial_guess = [0.5, 2, 30]
params = [0.9, -1] #beta = any, sigma = -1
roots, limit_cycle = limit_cycle_finder(ode(hopf,params),initial_guess,phase_condition)
#print(roots)
#print(limit_cycle)
fig = phase_portrait_plotter(limit_cycle)
fig.show()


# %%
t = np.linspace(0,10,100)

beta, sigma = params
theta = 1 
u1 = beta**0.5 * np.cos(t+theta)
u2 = beta**0.5 * np.sin(t+theta)

plt.plot(u1,u2)
plt.show()


# %%

plt.plot(u1,u2)
plt.plot(limit_cycle.y[0, :], limit_cycle.y[1, :], label='Isolated periodic orbit')
plt.show()

# %%
