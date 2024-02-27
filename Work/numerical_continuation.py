
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

def hopf(t,u,params):#params = [beta]
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

def natural_continuation(myode, u0, step_size, param_bounds):
    sol = [u0]
    param_values = np.arange(param_bounds[0],param_bounds[-1]+step_size,step_size)
    #For each perturbed parameter, approximate the solution using the previous solution as the initial guess, and append the solution to the list of solutions
    for param in param_values:
        #make an approximation of the solution
        u_tilde = sol[-1] 
        u_next = fsolve(lambda u: myode(0,u,param),u_tilde)
        sol.append(u_next)
    
    return np.array(sol), param_values




step_size = 0.01
u0 = np.array([1,1])
param_bounds = [0,2]

#u_list, parameter_list = cubic_continuation(cubic, u0, step_size, params)
sol, param_vals = natural_continuation(hopf, u0, step_size, param_bounds)

plt.plot(param_vals,sol[:-1,1])
plt.xlabel('Parameter')
plt.ylabel('Solution')
plt.title('Natural Continuation')
plt.show()









