
from ode_solver import solve_ode
from bvp_and_shooting import phase_condition, shoot, limit_cycle_finder
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from Equations_Functions import hopf, modified_hopf, lokta_volterra, hopf_bifurcation_3d, brusselator, hopfNormal
import warnings 
from numpy import linalg as LA
from scipy.optimize import root
import scipy

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



def numerical_continuation(f, method, x0, par_array, par_index, par_bounds, max_steps, discretization, solver, phase_condition, increase):

    
    def wrapper(par_array,u0):
        G = discretization(f,phase_condition)
        
        return solver(lambda u0: G(u0[:-1],u0[-1],par_array),u0)

    max_par, min_par = par_bounds[0], par_bounds[1]
    max_steps_PAL, max_steps_nat = max_steps[1], max_steps[0]

    if method == 'natural':
        sol, pars = natural_continuation(f, x0, max_steps_nat, par_bounds, phase_condition, wrapper)
        return pars, sol 
    elif method == 'pseudo':
        pars, sol = pseudo_continuation(f, x0, par_array, par_index, min_par, max_par, max_steps_PAL,wrapper,phase_condition,increase)
        return pars, sol
    else:
        raise ValueError("Invalid continuation method. Choose 'natural' or 'pseudo'.")



def natural_continuation(f,initial_guess,steps,param_bounds,phase_condition, wrapper):
    param_values = np.linspace(param_bounds[0],param_bounds[-1],steps)
    equilibria = np.zeros((len(param_values),len(initial_guess[:-1])))
    limit_cycle = np.zeros((len(param_values), len(initial_guess)))

    limit_cycle[0] = initial_guess
    equilibria[0] = initial_guess[:-1]
    guess = initial_guess[:-1] # guess
    prev_cycle = initial_guess

    if phase_condition == None:
        for index,par in enumerate(param_values[1:]):
            guess = equilibria[index]
            equilibrias = fsolve(lambda u:f(0,u,[par]), guess)
            equilibria[index+1] = equilibrias
        
        return equilibria, param_values
        #print(equilibrias)

    else:

        for index,par in enumerate(param_values[1:]):
            # sol = limit_cycle_finder(f,prev_cycle,phase_condition,[par],test=False)
            sol = wrapper([par],prev_cycle)
            limit_cycle[index+1] = sol 
            prev_cycle = sol
            #print(sol)
        return limit_cycle[1:,:], param_values[1:] 



def calculate_par_step(min_par, max_par, num_steps):
    return (max_par - min_par) / num_steps


def find_initial_sols(f, u0_guess, phase_condition,par_index, par0,par_step, wrapper):
    # Find the first solution
    if phase_condition == None:
        u1 = fsolve(lambda u:f(0,u,par0), u0_guess[:])
        u1 = np.append(u1,par0[par_index])
        
        par0[par_index] += par_step
        u2 = fsolve(lambda u:f(0,u,par0), u1[:-1])
        
        u2 = np.append(u2,par0[par_index]) #returns the equilibrium point [u1,u2,par]
    else:
        u1 = wrapper(par0,u0_guess)
        u1 = np.append(u1,par0[par_index])
        
        par0[par_index] += par_step
        # find the second solution
        u2 = wrapper(par0,u1[:-1])
        u2 = np.append(u2,par0[par_index])

    return u1, u2

def predict(u_current,u_previous):
    delta_u = u_current - u_previous #secant
    u_pred = u_current + delta_u 
    return u_pred, delta_u

def corrector_with_arclength(ode,u,par_index,par_array,u_pred,delta_u, u_old,par_step, phase_condition): #u is the var to solve for
   
    #secant:
    secant_u = delta_u[:-1] #u1,u2,u3,T
    secant_p = delta_u[-1] #par
    ds = LA.norm(secant_u**2 + secant_p**2)

    pAL = np.dot(u[:-1]-u_pred[:-1], secant_u) + np.dot(u[-1]-u_pred[-1], secant_p) 

    G = shoot(ode,phase_condition)
    
    if phase_condition == None:
        ode_res = ode(0,u[:-1],par_array)
        return np.append(ode_res, pAL)

    else:

        par_array[par_index] = u[-1]
        
        
        shoot_res =  G(u[:-2], u[-2],par_array)
        
        return np.append(shoot_res, pAL)



def pseudo_continuation(ode, x0, par_array, par_index, min_par, max_par, max_steps, wrapper, phase_condition, increase=False):
    """
    Parameters:
    - ode: The differential equation system to solve.
    - x0: Initial condition for the state variables.
    - par_array: Array of parameters.
    - par_index: The index of the parameter to continue in par_array.
    - min_par, max_par: The minimum and maximum bounds for the continuation parameter.
    - max_steps: The total number of continuation steps.
    - wrapper: Function that wraps the shooting method.
    - phase_condition: The phase condition for the system, can be None
    - increase: Flag indicating the direction of continuation. If True, increases the parameter; otherwise, decreases.
    """

    par_step = abs((max_par - min_par) / max_steps) * (1 if increase else -1) # Ensure the step is negative for decreasing
   
    
    #par_array[par_index] = max_par  # Start from the max parameter value
    par_array[par_index] = max_par if not increase else min_par
    u_old, u_current = find_initial_sols(ode, x0, phase_condition, par_index, par_array, par_step, wrapper)
    
    u_sol = [u_old[:-1]]  #, u_current[:-1]]
    alpha_sol = [u_old[-1]] #, u_current[-1]]

    #for i in range(1, max_steps+1):
    run = True

    while run:
        u_pred, delta_u = predict(u_current, u_old) #u_pred = [u1,u2,u3,T,par], delta_u = [du1,du2,du3,dT,dpar]
        updated_value = u_pred[-1]
        if phase_condition == None:
            correction_result = root(lambda u: corrector_with_arclength(ode, u, par_index, par_array, u_pred, delta_u, u_old, par_step,phase_condition), u_pred, method='lm',tol=1e-6)
            
        else:
            correction_result = root(lambda u: corrector_with_arclength(ode, u, par_index, par_array, u_pred, delta_u, u_old, par_step,phase_condition), u_pred, method='lm',tol=1e-6)
        
        u_corrected = correction_result.x
        

        if updated_value < min_par or updated_value > max_par:
            print("Parameter boundary reached or exceeded.")
            run = False
        else:
            # Update the parameter within the allowable range
            par_array[par_index] = updated_value
        
        u_old = u_current
        u_current = u_corrected
        
        u_sol.append(u_current[:-1])
        alpha_sol.append(u_current[-1])
        
    for i,sol in enumerate(u_sol):
        u_sol[i] = sol.tolist() # For plotting purposes, convert to list

    return alpha_sol, u_sol


def pseudo_plotter(par, sol):
    # Prepare lists to hold the individual components and norms
    components = [[] for _ in range(len(sol[0]))]  # Assuming all solution vectors are of the same length
    norms = []
    
    # Iterate over each solution vector
    for solution in sol:
        for i, component in enumerate(solution):
            components[i].append(component)
        # Compute the norm excluding the last component if there are more than one components
        norm = scipy.linalg.norm(solution[:-1] if len(solution) > 1 else solution, axis=0)
        norms.append(norm)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    # Plot the norm
    plt.plot(par, norms, label='Pseudo Arc Length', marker='o', linestyle='-')
    # Plot each component
    
    for i, component_data in enumerate(components[:-1]):  # Exclude the last component if it's time or similar
        plt.plot(par, component_data, label=f'u{i+1}', marker='.', linestyle='--')
    
    plt.xlabel('Parameter', fontsize=12)
    plt.ylabel('Solution Magnitude', fontsize=12)
    plt.title('Pseudo Arc Length Method', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

def natural_plotter(param_values, solutions):
    plt.figure(figsize=(12, 6))
    # Check if solutions is 1D or 2D
    if solutions.ndim == 1:
        # If 1D, plot directly
        plt.plot(param_values, solutions, label='Solution')
    else:
        # If 2D, plot as before
        for i in range(solutions.shape[1]-1):# Exclude the last component if it's time or similar
            plt.plot(param_values, solutions[:, i], label=f'u{i+1}')
    
    plt.xlabel('Parameter', fontsize=12)
    plt.ylabel('Solution Components', fontsize=12)
    plt.title('Natural Continuation Bifurcation Diagram', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()





def main():


    def phase_condition(ode,u0,pars):
    #return the phase condition which is du1/dt(0) = 0
        return np.array([ode(0,u0,pars)[0]])
    
    x0 = np.array([0.37, 3.5, 7.15])
    #x0 = np.array([1,1]) # For 1D systems use an extra IC because the phase condition is not needed
    par_array = [5]  # Start parameter value
    par_index = 0
    max_steps = [200, 30]
    phase_condition = phase_condition

    par_pseudo, sol_pseudo = numerical_continuation(brusselator, 'pseudo', x0, par_array, par_index, [3, 1.5], [200, 30], shoot, fsolve, phase_condition=phase_condition, increase=False)
    par_nat, sol_nat = numerical_continuation(brusselator, 'natural', x0, par_array, par_index, [3, 1.5], [200, 30], shoot, fsolve, phase_condition=phase_condition, increase=False)
    
    natural_plotter(par_nat[1:], sol_nat[1:])
    pseudo_plotter(par_pseudo, sol_pseudo)



if __name__ == "__main__":
   main()

#TODO: Clarify the variables used in the pseudo arc length method
