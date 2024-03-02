
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

def phase_condition(ode,u0,T):
    #return the phase condition which is du1/dt(0) = 0
    return np.array([ode(0,u0)[0]])

def natural_continuation(myode, u0, step_size, param_bounds, phase_condition):
    param_values = np.arange(param_bounds[0],param_bounds[-1]+step_size,step_size)
    sol = np.zeros(((len(param_values),len(u0))))
    #first sol:
    roots,_ = limit_cycle_finder(ode(myode, param_values[0]), u0, phase_condition)
    sol[0] = roots
    #Now use this known solution to find the next 
    u_tilde = roots
    for index,param in enumerate(param_values[1:]):
        perturbed_roots,_ = limit_cycle_finder(ode(myode, param), u_tilde, phase_condition)
        sol[index+1] = perturbed_roots
        u_tilde = perturbed_roots #could use a linear combination of the previous solution and the perturbed solution

    return sol[:,:], param_values[:]

    #For each perturbed parameter, approximate the solution using the previous solution as the initial guess, and append the solution to the list of solutions
    #Need to find a solution using shooting algorithm and limit-cycle finder
    #Watch the video
    #for param in param_values:



step_size = 0.05
u0 = np.array([-2,2,5])
param_bounds = [0,2]


#u_list, parameter_list = cubic_continuation(cubic, u0, step_size, params)
sol, param_values = natural_continuation(hopf, u0, step_size, param_bounds, phase_condition)

# plt.plot(param_values,sol[:,0])
# plt.plot(param_values,sol[:,1])
# plt.plot(param_values,sol[:,2])
#What value of Sol should i plot?
#Sol array = [u1,u2,T] for each parameter value : 3xN array, what value of sol do i plot?
# Subplot of all sol

fig, ax = plt.subplots(3,1)
ax[0].plot(param_values,sol[:,0])
ax[1].plot(param_values,sol[:,1])
ax[2].plot(param_values,sol[:,2])
print(sol)
plt.show()










