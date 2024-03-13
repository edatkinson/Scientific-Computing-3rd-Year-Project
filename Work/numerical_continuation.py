
from new_ode_solver import solve_ode
#from bvp_and_shooting import phase_condition, shoot, limit_cycle_finder
from mybvp import phase_condition, shoot, limit_cycle_finder
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from Equations_Functions import hopf, modified_hopf
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



def main():
    
    # steps = 20
    # initial_guess = np.array([1, 1.0, 4])    #np.array([0.2,0.5,35]) works!
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
    max_par = -1
    par_step = 100
    initial_par = 2
    sol =  pseudo_method(modified_hopf, np.array([1.41, 0, 6.28,2]), initial_par, phase_condition, max_par,par_step)
    print(sol)
    plt.plot(sol[:,3], sol[:,0], label='u0')
    plt.plot(sol[:,3], sol[:,1], label='u1')
    plt.legend()
    plt.show()




def plotter(param_values, cycles, eq):
    plt.figure(figsize=(8, 6))
    plt.plot(param_values, cycles[:,2], label='T')
    plt.xlabel('Parameter', fontsize=12)
    plt.ylabel('T', fontsize=12)
    plt.title('Bifurcation of T', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(param_values, cycles[:,0], label='u1')
    plt.plot(param_values, cycles[:,1], label='u2')
    plt.xlabel('Parameter', fontsize=12)
    plt.ylabel('u', fontsize=12)
    plt.title('Bifurcation of u1 and u2', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(param_values, eq[:,0], label='u1')
    plt.plot(param_values, eq[:,1], label='u2')
    plt.xlabel('Parameter', fontsize=12)
    plt.ylabel('u', fontsize=12)
    plt.title('Equilibria', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
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
        equilibrias = fsolve(lambda u:f(0,u,par), guess)
        equilibria[index+1] = equilibrias
        #print(equilibrias)

    for index,par in enumerate(param_values[1:]):
        sol, cycle = limit_cycle_finder(f,prev_cycle,phase_condition,par)
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

def secant(v0,v1):
    return v1 - v0  

def pseudo_arclength_equation(u0,u1,par0,par1):
    # par0 = np.array(par0).flatten()
    # par1 = np.array(par1).flatten()
    # print(par0,par1)
    v0 = np.append(u0[:-1],par0)
    v1 = np.append(u1[:-1],par1)
    #print(v0,v1)

    delta = secant(v0,v1)

    approx = v1 - secant(v0, v1)

    return approx, np.dot(secant(v0,v1), approx)


def pseudo_method(f, u0_guess, initial_par,phase_condition,max_par,par_step): #guesses: [u,T]
    u0 = u0_guess # [u,T,param]
    par_1 = initial_par
    
    par_list = np.linspace(par_1,max_par,par_step) #List of parameter to iterate over

    u1,_ = limit_cycle_finder(f,u0[:-1],phase_condition,par_1)

    u2,_ = limit_cycle_finder(f,u1,phase_condition,par_list[1])

    u1 = np.append(u1,par_1)
    u2 = np.append(u2,par_list[1])
    
    sol = np.zeros((len(par_list)+1,len(u0_guess)))
    sol[0] = u1
    sol[1] = u2
    G = shoot(f,phase_condition)

    for index,par in enumerate(par_list[1:]):
        approx, pseudo_arc_length = pseudo_arclength_equation(sol[index],sol[index+1], par_list[index], par_list[index+1])

        G_o = G(sol[index][:-2], sol[index][-2],par) #G = 

        f_new = np.hstack((G_o, pseudo_arc_length))

        correct_sol = fsolve(lambda nu: f_new, approx, xtol=1e-6, epsfcn=1e-6)

        sol[index+2] = correct_sol

    return sol


if __name__ == "__main__":
   main()


