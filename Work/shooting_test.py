
# %%
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import bvp_and_shooting 

# %%

initial_guess = [0.5, 2, 30]
params = [0.9, -1] #beta = any, sigma = -1
roots, limit_cycle = bvp_and_shooting.limit_cycle_finder(bvp_and_shooting.ode(bvp_and_shooting.hopf,params),initial_guess,bvp_and_shooting.phase_condition)
#print(roots)
fig = bvp_and_shooting.phase_portrait_plotter(limit_cycle)
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
