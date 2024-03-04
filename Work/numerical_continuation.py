
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

def cubic(t,x, param):
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


def nat_continuation(myode,initial_guess, step_size, param_bounds, phase_condition):
    equilibria = [initial_guess[:-1]]
    param_values = np.arange(param_bounds[0],param_bounds[-1]+step_size,step_size)
    for param in param_values:
        guess = equilibria[-1]
        equilibrias = fsolve(lambda u: myode(0,u,param), guess)
        equilibria.append(equilibrias)

    return np.array(equilibria[1:]), param_values








#For each perturbed parameter, approximate the solution using the previous solution as the initial guess, and append the solution to the list of solutions
#Need to find a solution using shooting algorithm and limit-cycle finder
#Watch the video
#for param in param_values:

#(1)Need 2 known solutions, use shooting to find these based off of two different initial guesses
#(2) Generate a secant: Delta = v(i) - v(i-1)
#(3) Predict the Solution: v(i+1) = v(i) + Delta
#(4) Stack the pseudo arc length equation 

def secant(v0,v1):
    return v1 - v0

def psuedo_arc_length_eq(sol_1,sol_2): #sol_1 = [u_0,u_1,T_1, param1], sol_2 = [u_2,u_3,T_2, param2]
    v0 = np.array([sol_1])
    v1 = np.array([sol_2])

    delta = np.squeeze(secant(v0,v1))
    approx = np.squeeze(v1 + delta)  

    return np.dot(delta, approx)

def pseudo_method(myode,current,guess,phase_condition): #current = [u_0,u_1,T_1, param_1], guess = [u_2,u_3,T_2, param_2]
    def augmented_system(f, current,guess,phase_condition):
        estimate_1 = guess[:-1]
        param = guess[-1]
        return np.hstack((shoot(ode(f,param), estimate_1,phase_condition), psuedo_arc_length_eq(current,guess)))

    corrected_sol = fsolve(lambda U: augmented_system(myode,current,U,phase_condition),guess)
    return corrected_sol #This is the known solution with the limit cylce inducing parameter




#Write a function which displays how the limit cycle changes as the parameter changes
#Use the pseudo arc length method to find the next solution
#Repeat until the parameter range is exhausted
#Plot the solutions
#Plot the bifurcation diagram




#sol = pseudo_method(hopf,[0,1,5,1.5],[1,0,5,2],phase_condition)


# 1. Find the first two solutions using the shooting algorithm
# 2. Use the pseudo arc length method to find the next solution
# 3. Repeat until the parameter range is exhausted
# 4. Plot the solutions
# 5. Plot the bifurcation diagram




step_size = 0.05
initial_guess = np.array([1,1,3])
param_bounds = [0,2]


#u_list, parameter_list = cubic_continuation(cubic, u0, step_size, params)
sol, params = nat_continuation(hopf, initial_guess, step_size, param_bounds, phase_condition)
print(sol)
plt.plot(params,sol)
plt.show()


# plt.plot(param_values,sol[:,0])
# plt.plot(param_values,sol[:,1])
# plt.plot(param_values,sol[:,2])
#What value of Sol should i plot?
#Sol array = [u1,u2,T] for each parameter value : 3xN array, what value of sol do i plot?
# Subplot of all sol

# fig, ax = plt.subplots(3,1)
# ax[0].plot(param_values,sol[:,0])
# ax[1].plot(param_values,sol[:,1])
# ax[2].plot(param_values,sol[:,2])
# print(sol)
# plt.show()










