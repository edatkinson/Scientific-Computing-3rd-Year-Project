
from new_ode_solver import solve_ode
from bvp_and_shooting import phase_condition, shoot, limit_cycle_finder
#from mybvp import phase_condition, shoot, limit_cycle_finder
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from Equations_Functions import hopf, modified_hopf, lokta_volterra
import warnings 
from numpy import linalg as LA

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




def plotter(param_values, cycles, eq):
    # plt.figure(figsize=(8, 6))
    # plt.plot(param_values, cycles[:,2], label='T')
    # plt.xlabel('Parameter', fontsize=12)
    # plt.ylabel('T', fontsize=12)
    # plt.title('Bifurcation of T', fontsize=14)
    # plt.legend(fontsize=12)
    # plt.grid(True)
    # plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(param_values, cycles[:,0], label='u1')
    plt.plot(param_values, cycles[:,1], label='u2')
    plt.xlabel('Parameter', fontsize=12)
    plt.ylabel('u', fontsize=12)
    plt.title('Natural Continuation Bifurcation Diagram of u1 and u2 ', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.plot(param_values, eq[:,0], label='u1')
    # plt.plot(param_values, eq[:,1], label='u2')
    # plt.xlabel('Parameter', fontsize=12)
    # plt.ylabel('u', fontsize=12)
    # plt.title('Equilibria', fontsize=14)
    # plt.legend(fontsize=12)
    # plt.grid(True)
    # plt.show()



def cubic(t,x, param):
    f = x**3 - x - param
    return f

def hopf_example(t,u,pars):#params = (beta)
    beta = pars
    du1dt = beta*u[0] - u[1] - u[0] * ((u[0])**2 + (u[1])**2)
    du2dt = u[0] + beta*u[1] - u[1] * ((u[0])**2 + (u[1])**2) 
    dUdt = np.array([du1dt,du2dt])
    return dUdt

def wrapper(x,pars):
    return modified_hopf(0,x,pars)

def natural_continuation(f,initial_guess,steps,param_bounds,phase_condition):
    param_values = np.linspace(param_bounds[0],param_bounds[-1],steps)
    equilibria = np.zeros((len(param_values),len(initial_guess[:-1])))
    limit_cycle = np.zeros((len(param_values), len(initial_guess)))

    limit_cycle[0] = initial_guess
    equilibria[0] = initial_guess[:-1]
    guess = initial_guess[:-1]
    prev_cycle = initial_guess

    for index,par in enumerate(param_values[1:]):
        guess = equilibria[index]
        equilibrias = fsolve(lambda u:f(0,u,[par]), guess)
        equilibria[index+1] = equilibrias
        #print(equilibrias)

    for index,par in enumerate(param_values[1:]):
        sol, cycle = limit_cycle_finder(f,prev_cycle,phase_condition,[par])
        limit_cycle[index+1] = sol 
        prev_cycle = sol
        #print(prev_cycle)
    #Could combine loops?

    return limit_cycle[1:,:], param_values[1:], equilibria[1:,:]





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


def calculate_par_step(min_par, max_par, num_steps):
    return (max_par - min_par) / num_steps


def find_initial_sols(f, u0_guess, phase_condition,par_index, par0,par_step):
    #par0 = par0[0]
    u1,_ = limit_cycle_finder(f,u0_guess,phase_condition,par0)
    u1 = np.append(u1,par0[par_index])
    
    par0[par_index] += par_step
    u2,_ = limit_cycle_finder(f,u1[:-1],phase_condition,par0)
    u2 = np.append(u2,par0[par_index])
    
    return u1, u2



def predict(u_current,u_previous):
    delta_u = u_current - u_previous
    u_approx = u_current + delta_u
    return [u_approx, delta_u]

def corrector(ode,u,par_index,par_array,u_pred,delta_u): #u is the var to solve for
    par_array[par_index] = u[-1]
    G = shoot(ode,phase_condition)
    shoot_res =  G(u[:-2], u[-2],par_array)
    pAL = np.dot(u - u_pred, delta_u)
    return np.append(shoot_res, pAL)


# Adjusting the continuation function for a parameter range from 3 to 2
def pseudo_continuation(ode, x0, par_array, par_index, min_par, max_par, max_steps):
    par_step = -abs((max_par - min_par) / max_steps)  # Ensure the step is negative for decreasing
    
    sol = np.zeros((len(x0) + 1, max_steps + 1))
    
    par_array[par_index] = max_par  # Start from the max parameter value
    u_old, u_current = find_initial_sols(ode, x0, phase_condition, par_index, par_array, par_step)
    sol[:, 0] = u_old
    
    for i in range(1, max_steps + 1):
        if par_array[par_index] + par_step < min_par:
            print("Parameter boundary reached or exceeded.")
            break
        
        par_array[par_index] += par_step
        #print(par_array[par_index])
        u_pred, delta_u = predict(u_current, u_old)
        
        u_corrected = fsolve(lambda u: corrector(ode, u, par_index, par_array, u_pred, delta_u), u_pred, xtol=1e-6, epsfcn=1e-6)
        u_corrected = np.append(u_corrected[:-1], par_array[par_index])  # Ensure the last value is the parameter
        
        u_old = u_current
        u_current = u_corrected
        sol[:, i] = u_current
        #print(sol[:, i])
    return sol[:, :i]


def main():
    
    # steps = 20
    # initial_guess = np.array([1, 1.0, 4])    #np.array([0.2,0.5,35]) works with mybvp
    # param_bounds = [2,-1]

    # limit_cycle, param_values, eq = natural_continuation(
    #     modified_hopf, 
    #     initial_guess, 
    #     steps, 
    #     param_bounds, 
    #     phase_condition)

    # # #Bifurcation diagram of Limit cycle and equilibria
    # plotter(param_values, limit_cycle, eq)


    # ####Pseudo Arc Length Method #####

    max_steps = 100
    x0 = np.array([3.4, 3.2, 6.2])
    par_array = [2]  # Start parameter value
    par_index = 0
    par_step = -0.1  # Ensure the step is negative for decreasing
    min_par = -1
    max_par = 2

    sol = pseudo_continuation(modified_hopf, x0, par_array, par_index, min_par, max_par, max_steps)
    print(sol)
    plt.plot(sol[3, :], sol[0, :], label='u1')
    plt.plot(sol[3, :], sol[1, :], label='u2')
    plt.xlabel('Parameter', fontsize=12)
    plt.xlim(min_par, max_par)
    plt.ylabel('u', fontsize=12)
    plt.title('Pseudo Arc Length Method', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
   main()


