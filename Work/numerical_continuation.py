
from new_ode_solver import solve_ode
from bvp_and_shooting import phase_condition, shoot, limit_cycle_finder
#from mybvp import phase_condition, shoot, limit_cycle_finder
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from Equations_Functions import hopf, modified_hopf, lokta_volterra, hopf_bifurcation_3d, brusselator, hopfNormal
import warnings 
from numpy import linalg as LA
from scipy.optimize import root

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


def plotter(param_values, cycles, eq, sol):
    # plt.figure(figsize=(8, 6))
    # plt.plot(param_values, cycles[:,2], label='T')
    # plt.xlabel('Parameter', fontsize=12)
    # plt.ylabel('T', fontsize=12)
    # plt.title('Bifurcation of T', fontsize=14)
    # plt.legend(fontsize=12)
    # plt.grid(True)
    # plt.show()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(param_values, cycles[:,0], label='u1')
    plt.plot(param_values, cycles[:,1], label='u2')
    #plt.plot(param_values, cycles[:,2], label='u3')
    plt.xlabel('Parameter', fontsize=12)
    plt.ylabel('u', fontsize=12)
    plt.title('Natural Continuation Bifurcation Diagram of u1, u2, and u3', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(sol[4-1, :], sol[0, :], label='x')
    plt.plot(sol[4-1, :], sol[1, :], label='y')
    #plt.plot(sol[4-1, :], sol[2, :], label='z')
    plt.xlabel('Parameter', fontsize=12)
    # plt.xlim(min_par, max_par)
    plt.ylabel('u', fontsize=12)
    plt.title('Pseudo Arc Length Method', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
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
        sol, cycle = limit_cycle_finder(f,prev_cycle,phase_condition,[par],test=False)
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
    u1,_ = limit_cycle_finder(f,u0_guess,phase_condition,par0,test=False)
    u1 = np.append(u1,par0[par_index])
    
    par0[par_index] += par_step
    u2,_ = limit_cycle_finder(f,u1[:-1],phase_condition,par0,test=False)
    u2 = np.append(u2,par0[par_index])

    return u1, u2



def predict(u_current,u_previous):
    delta_u = u_current - u_previous #secant
    u_pred = u_current + delta_u 
    return u_pred, delta_u

def corrector_with_arclength(ode,u,par_index,par_array,u_pred,delta_u, u_old,par_step): #u is the var to solve for
    par_array[par_index] = u[-1]
    #print(u[-1])
    G = shoot(ode,phase_condition)
    shoot_res =  G(u[:-2], u[-2],par_array)
    
    #secant:
    secant_u = delta_u[:-1] #u1,u2,u3,T
    secant_p = delta_u[-1] #par

    # print(u_pred, 'u_pred')
    # print(u_old, 'u_old')
    # print(u, 'u')
    ds = LA.norm(secant_u**2 + secant_p**2)

    pAL = np.dot(u[:-1]-u_pred[:-1], secant_u) + np.dot(u[-1]-u_pred[-1], secant_p) 
    #print(pAL, 'pAL')

    return np.append(shoot_res, pAL)



def pseudo_continuation(ode, x0, par_array, par_index, min_par, max_par, max_steps,increase=False):
    """
    Parameters:
    - ode: The differential equation system to solve.
    - x0: Initial condition for the state variables.
    - par_array: Array of parameters.
    - par_index: The index of the parameter to continue in par_array.
    - min_par, max_par: The minimum and maximum bounds for the continuation parameter.
    - max_steps: The total number of continuation steps.
    - increase: Flag indicating the direction of continuation. If True, increases the parameter; otherwise, decreases.
    """

    par_step = abs((max_par - min_par) / max_steps) * (1 if increase else -1) # Ensure the step is negative for decreasing
   
    #sol = np.zeros((len(x0) + 1), max_steps + 1)
    
    #par_array[par_index] = max_par  # Start from the max parameter value
    par_array[par_index] = max_par if not increase else min_par
    u_old, u_current = find_initial_sols(ode, x0, phase_condition, par_index, par_array, par_step)
    # sol = np.append(sol,u_old) # u_old = [u1,u2,u3,T,par]
    #sol[:,0] = u_old
    u_sol = [u_old[:-1]]  #, u_current[:-1]]
    alpha_sol = [u_old[-1]] #, u_current[-1]]

    #for i in range(1, max_steps+1):
    run = True
    i = 0
    while run:
        u_pred, delta_u = predict(u_current, u_old) #u_pred = [u1,u2,u3,T,par], delta_u = [du1,du2,du3,dT,dpar]
        updated_value = u_pred[-1]
        
         
        #u_corrected = fsolve(lambda u: corrector_with_arclength(ode, u, par_index, par_array, u_pred, delta_u, u_old, par_step), u_pred)#, xtol=1e-6, epsfcn=1e-6)
        correction_result = root(lambda u: corrector_with_arclength(ode, u, par_index, par_array, u_pred, delta_u, u_old, par_step), u_pred, method='lm',tol=1e-6)
        u_corrected = correction_result.x
        #print(u_corrected, 'af')

        if updated_value < min_par or updated_value > max_par:
            print("Parameter boundary reached or exceeded.")
            run = False
        else:
            # Update the parameter within the allowable range
            par_array[par_index] = updated_value
        
        u_old = u_current
        u_current = u_corrected
        
        i += 1
        print(u_current)
        u_sol.append(u_current[:-1])
        alpha_sol.append(u_current[-1])
        
        #sol[:, i] = u_current

        # print(sol[:, i])

    return alpha_sol, u_sol

# def calculate_par_step(min_par, max_par, num_steps):
#     return (max_par - min_par) / num_steps


# def find_initial_sols(f, u0_guess, phase_condition,par_index, par_array, par_step, vary_par_range):
#     par0 = par_array[0]
#     #par_step *= np.sign(vary_par_range[1] - vary_par_range[0])
#     #print(u0_guess, 'u0_guess')

#     u1,_ = limit_cycle_finder(f,u0_guess,phase_condition,[par0])
#     u1 = np.append(u1,par0)
    
#     par0 += par_step
#     #print(par0, 'par0' )
#     u2,_ = limit_cycle_finder(f,u1[:-1],phase_condition,[par0])
#     u2 = np.append(u2,par0)
#     return u1, u2


    


# def pseudo_continuation(f,x0,par_array,vary_par,par_range,steps):

#     '''
#     params:
#     f: function to solve
#     u0: initial guess
#     par_array: array of parameters
#     vary_par: index of parameter to vary
#     range: range of parameter to vary [max,min]
#     steps: number of steps
#     '''
#     def predict(u_current,u_previous):
#         delta_u = u_current - u_previous #secant
#         u_pred = u_current + delta_u 
#         return u_pred, delta_u

#     def corrector_with_arclength(ode,u,par_index,par_array,u_pred,delta_u, u_old): #u is the var to solve for
#         par_array[par_index] = u[-1]
#         #print(u[-1])
#         G = shoot(ode,phase_condition)
#         shoot_res =  G(u[:-2], u[-2],par_array)
        
#         #secant:
#         secant_u = delta_u[:-1] #u1,u2,u3,T
#         secant_p = delta_u[-1] #par
#         # print(secant_p, 'secant_p')
#         # print(secant_u, 'secant_u')
        
#         du =  u_pred[:-1]
#         dp =  u_pred[-1]

#         # print(du, 'u_pred')
#         # print(dp, 'p_pred')

#         # print(u_pred, 'u_pred')
#         # print(u_old, 'u_old')
#         # print(u, 'u')
#         ds = LA.norm(secant_u**2 + secant_p**2)

#         pAL = np.dot(u[:-1]-u_pred[:-1], secant_u) + np.dot(u[-1]-u_pred[-1], secant_p) 
#         #print(pAL, 'pAL')

#         return np.append(shoot_res, pAL)


#     min_par = par_range[1]
#     max_par = par_range[0]

#     par_step = calculate_par_step(par_range[1],par_range[0],steps)
#     #par_step = 0.015
#     par_step *= np.sign(min_par - max_par) 
#     #print(par_step)
#     sol = np.zeros((len(x0) + 1, steps + 1))

#     u_old, u_current = find_initial_sols(f, x0, phase_condition, vary_par, par_array, par_step,par_range)
#     #print(u_old, 'u_old')
#     #print(u_current, 'u_current')
#     sol[:, 0] = u_old # u_old = [u1,u2,u3,T,par]

#     par_array[vary_par] = max_par #setting param value in par_array to max_par

#     for i in range(1, steps+1):
#         u_pred, delta_u = predict(u_current, u_old) #u_pred = [u1,u2,u3,T,par], delta_u = [du1,du2,du3,dT,dpar]
#         par_array[vary_par] = u_pred[-1]

#         #print(par_array[vary_par])
#         if par_array[vary_par] < min_par or par_array[vary_par] > max_par:
#             print("Parameter boundary reached or exceeded.")
#             break
#         else:
#             # Update the parameter within the allowable range
#             par_array[vary_par] = u_pred[-1]

#         # print(u_pred[:-1], 'u_pred')
#         # print(u_pred[-1], 'p_pred')
#         # print(delta_u[:-1], 'secant_u')
#         # print(delta_u[-1], 'p_secant')


#         correction_result = root(lambda u: corrector_with_arclength(f, u, vary_par, par_array, u_pred, delta_u, u_old), u_pred, method='lm')
        
#         u_corrected = correction_result.x
#         #print(u_corrected, 'af')
#         u_old = u_current
#         u_current = u_corrected
#         #print(u_current, 'u_current')
#         #print(u_old, 'u_old')
#         sol[:, i] = u_current
#         print(sol[:, i])
#     return sol[:, :i]



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

    max_steps = 200
    x0 = np.array([1.04, 0.52, -0.52, 10])
    #
    #x0 = np.array([0, 1, 5])
    #x0 = np.array([3.4, 3.2, 6.2])
    par_array = [2]
    par_index = 0
    min_par = -1
    max_par = 2

    # lim,par, eq = natural_continuation(modified_hopf, x0, 200, [2,-1], phase_condition)
    # print('natty done')
    import scipy
    
    par, u_sol = pseudo_continuation(hopf_bifurcation_3d, x0, par_array, par_index, min_par, max_par, max_steps, increase=False)
    
    par_range = [2,-1]
    vary_par = 0
    steps = 100
    #print(sol)
    #sol = pseudo_continuation(modified_hopf,x0,par_array,vary_par,par_range,steps)

    u1,u2,u3 = [], [], []
    count = 0
    while count < len(u_sol):
        u1.append(u_sol[count][0])
        u2.append(u_sol[count][1])
        u3.append(u_sol[count][2])
        u_sol[count] = np.array(scipy.linalg.norm(u_sol[count][:-1], axis=0, keepdims=True)) #exclding the T value
        count += 1
    
    
    plt.plot(par, u_sol, label='Pesudo Arc Length')
    plt.plot(par, u1, label='u1')
    plt.plot(par, u2, label='u2')
    plt.plot(par, u3, label='u3')
    plt.xlabel('Parameter', fontsize=12)
    plt.ylabel('||x||', fontsize=12)
    plt.title('Pseudo Arc Length Method', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


    
    # plt.plot(par, LA.norm(lim[:,0:1],axis=1),label='natural continuation')
    # plt.xlabel('beta')
    # plt.ylabel('||x||')
    # plt.legend()
    # plt.show()

    # plotter(par, lim, eq, sol)

    


if __name__ == "__main__":
   main()

#TODO: Clarify the variables used in the pseudo arc length method
