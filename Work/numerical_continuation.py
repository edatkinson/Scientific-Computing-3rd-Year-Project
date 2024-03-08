
from odesolver import solve_ode
from bvp_and_shooting import phase_condition, shoot, limit_cycle_finder
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import fsolve


from Equations_Functions import hopf, modified_hopf

import warnings 

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



def main():
    
    steps = 5000
    initial_guess = np.array([2,1,5])
    param_bounds = [0,2]
    
    sol,params,cycles = natural_continuation(
        hopf_example, 
        initial_guess, 
        steps, 
        param_bounds, 
        phase_condition)
    
    #sol, params, cycles 
    #print(sol) Equilibria of hopf is always 0,0 - why?

   # Bifurcation Diagram: 
    plotter(params, cycles)

    # ####Pseudo Arc Length Method #####

    # sol_1 = np.array([1,1,5,1.5])
    # sol_2 = np.array([1,1.1,5,1.5])
    # pseudo_arc_length_eq(sol_1, sol_2, 0.1)
    # correct_sol = pseudo_method(hopf_example,sol_1,sol_2,phase_condition)
    # print(sol)





def plotter(params, cycles, eq):
    plt.plot(param_values, cycles[:,2], label='T')
    plt.legend()
    plt.show()

    plt.plot(param_values, cycles[:,0], label='u1')
    plt.plot(param_values, cycles[:,1], label='u2')
    plt.xlabel('Parameter')
    plt.ylabel('u')
    plt.legend()
    plt.show()


def cubic(t,x, param):
    f = x**3 - x - param
    return f

def hopf_example(t,u,pars):#params = (beta)
    beta = pars
    du1dt = beta*u[0] - u[1] - u[0] * ((u[0])**2 + (u[1])**2)
    du2dt = u[0] + beta*u[1] - u[1] * ((u[0])**2 + (u[1])**2) 
    dUdt = np.array([du1dt,du2dt])
    return dUdt


def secant(v0,v1):
    return v1 - v0

def wrapper(x,pars):
    return modified_hopf(0,x,pars)

def natural_continuation2(f,initial_guess,steps,param_bounds,phase_condition):
    param_values = np.linspace(param_bounds[0],param_bounds[-1],steps)
    equilibria = np.zeros((len(param_values),len(initial_guess[:-1])))
    limit_cycle = np.zeros((len(param_values), len(initial_guess)))
    limit_cycle[0] = initial_guess
    equilibria[0] = initial_guess[:-1]
    guess = initial_guess[:-1]
    prev_cycle = initial_guess
    for index,par in enumerate(param_values[1:]):
        guess = equilibria[index]
        equilibrias = fsolve(lambda u:modified_hopf(0,u,par), guess)
        equilibria[index+1] = equilibrias
        print(equilibrias)

    for index,par in enumerate(param_values[1:]):
        sol, cycle = limit_cycle_finder(f,prev_cycle,phase_condition,par)
        limit_cycle[index+1] = sol 

        if len(limit_cycle[1,:]) > 3:
            cycle1 = limit_cycle[index-1]
            cycle2 = limit_cycle[index]
            #diff = cycle2 - cycle1
            guess = 2*prev_cycle - limit_cycle[index-2] 
            prev_cycle = guess
            print(prev_cycle)
        else:
            prev_cycle = sol 

    return limit_cycle[1:,:], param_values[1:], equilibria


limit_cycle, param_values, eq = natural_continuation2(modified_hopf, [1.4,1,30], 1000, [2,-1], phase_condition)
plotter(param_values, limit_cycle, eq)





#sol = pseudo_method(hopf,[0,1,5,1.5],[1,0,5,2],phase_condition)


# 1. Find the first two solutions using the shooting algorithm
# 2. Use the pseudo arc length method to find the next solution
# 3. Repeat until the parameter range is exhausted
# 4. Plot the solutions
# 5. Plot the bifurcation diagram


#(1)Need 2 known solutions, use shooting to find these based off of two different initial guesses
#(2) Generate a secant: Delta = v(i) - v(i-1)
#(3) Predict the Solution: v(i+1) = v(i) + Delta
#(4) Stack the pseudo arc length equation 



def pseudo_arc_length_eq(sol_1, sol_2, s): #sol1 = array[u_0,u_1,T_1, param1], sol2 = [u_2,u_3,T_2, param2]
    v0 = np.array([sol_1])
    v1 = np.array([sol_2])

    delta = np.squeeze(secant(v0,v1))
    approx = np.squeeze(v1 + delta)  

    return np.dot(delta, approx)

def pseudo_method(myode,current,guess,phase_condition): #current = [u_0,u_1,T_1, param_1], guess = [u_2,u_3,T_2, param_2]
    def augmented_system(U):
        estimate_1 = U[:-1]
        param = U[-1]

        shooting = shoot(ode(myode,param), estimate_1,phase_condition)
        print(shooting)
        pal = pseudo_arc_length_eq(current,guess,0.1)
        print(pal)
        return np.hstack((shooting, pal))

    corrected_sol = fsolve(augmented_system,guess)
    return corrected_sol #This is the known solution with the limit cylce inducing parameter


# if __name__ == "__main__":
#     main()




