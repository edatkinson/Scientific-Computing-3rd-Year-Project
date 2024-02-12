
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

