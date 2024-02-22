
from odesolver import solve_ode
from bvp_and_shooting import integrate, ode, phase_condition, shoot, limit_cycle_finder
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

'''
Example of a good interface
results = continuation(
    myode,
    x0, # initial state
    par_to_vary, # parameter to vary
    step_size, # step size
    max_steps, # maximum number of steps
    discretization=discretization, # discretization scheme
    root_finder=scipy.optimize.fsolve # solver to use
)
'''

def cubic(x, param):
    f = x**3 - x - param
    return f

def hopf(t,u,params: list):#params = [beta]
    beta = params
    du1dt = beta*u[0] - u[1] - u[0] * ((u[0])**2 + (u[1])**2)
    du2dt = u[0] + beta*u[1] - u[1] * ((u[0])**2 + (u[1])**2) 
    dUdt = np.array([du1dt,du2dt])
    return dUdt


def cubic_continuation(ode,u0,step_size,param_bounds): #Uses natural continuation on the cubic function
    roots = []
    param_values = np.arange(param_bounds[0],param_bounds[-1]+step_size,step_size)
    for param in param_values:
        root = fsolve(cubic, u0, args=(param,))
        roots = np.append(roots, root[0])
        u0 = root
    return roots, param_values

def natural_continuation(ode, u0, step_size, param_bounds):
    roots = np.array(u0)
    param_values = np.arange(param_bounds[0],param_bounds[-1]+step_size,step_size)
    for param in param_values:
        root = fsolve(ode, u0, args=(param,))
        roots = np.append(roots, root[0])
        u0 = root

u0 = np.array([0])
step_size = 0.01
params = [-2,2]

u_list, parameter_list = cubic_continuation(cubic, u0, step_size, params)

plt.plot(parameter_list, u_list)
plt.xlabel('Parameter')
plt.ylabel('Solution')
plt.title('Natural Continuation')
plt.show()







