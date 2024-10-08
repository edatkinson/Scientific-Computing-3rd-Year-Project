
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
from sparse_dense_bvp import solve_equation


def numerical_continuation(f, method, x0, par_array, par_index, par_bounds, max_steps, discretization, solver, phase_condition, increase):
    """
    Conducts numerical continuation for a given system of equations.

    Args:
        f: The function defining the system of equations.
        method (str): The continuation method ('natural' or 'pseudo').
        x0 (np.ndarray): Initial guess for the variables.
        par_array (list): List of parameters for the system.
        par_index (int): Index of the parameter to continue.
        par_bounds (tuple): Tuple of (Max, Min) bounds for the parameter.
        max_steps (tuple): Maximum steps for continuation.
        discretization: Discretization function.
        solver: Solver function.
        phase_condition: Function defining the phase condition.
        increase (bool): Direction of continuation.

    Returns:
        pars: List of parameter values.
        sol: List of solutions.
    """
    max_par, min_par = par_bounds
    max_steps_PAL, max_steps_nat = max_steps

    def wrapper(par_array, u0):
        G = discretization(f, phase_condition)
        return solver(lambda u0: G(u0[:-1], u0[-1], par_array), u0)

    if method == 'natural':
        pars, sol = natural_continuation(f, x0, max_steps_nat, par_bounds,par_array,par_index, phase_condition, wrapper)
        return pars, sol 
    elif method == 'pseudo':
        pars, sol = pseudo_continuation(f, x0, par_array, par_index, min_par, max_par, max_steps_PAL, wrapper, phase_condition, increase)
        return pars, sol
    else:
        raise ValueError("Invalid continuation method. Choose 'natural' or 'pseudo'.")



def natural_continuation(f, initial_guess, steps, param_bounds,par_array,par_index, phase_condition, wrapper):
    """
    Performs natural continuation on a system to track equilibria or limit cycles across a range of parameter values.

    Args:
        f (callable): The function defining the system of equations, expected to take the form f(t, u, params).
        initial_guess (np.ndarray): Initial guess for the state variables.
        steps (int): Number of steps to divide the parameter range into.
        param_bounds (tuple): Tuple of (min, max) bounds for the parameter to be varied.
        phase_condition (callable or None): Function enforcing a phase condition, or None if not applicable.
        wrapper (callable): A wrapper function that applies the corrector method and handles phase conditions.

    Returns:
        tuple: (param_values, equilibria) where param_values are the parameter values and equilibria are the corresponding states.
               Returns (None, None) if an error occurs.

    Raises:
        Exception: General exceptions related to solving the system or applying the wrapper, with warnings for specific failures.
    """
    try:
        par_array[par_index] = param_bounds[0]
        param_values = np.linspace(param_bounds[0], param_bounds[1], steps)
        equilibria = np.zeros((len(param_values), len(initial_guess)))

        equilibria[0] = initial_guess
        guess = initial_guess  # Initial guess

        if phase_condition is None:
            for index, par in enumerate(param_values[1:], start=0):
                guess = equilibria[index]
                try:
                    par_array[par_index] = par
                    equilibria[index+1] = fsolve(lambda u: f(0, u, par_array), guess)
                except Exception as e:
                    warnings.warn(f"Failed to find equilibrium at parameter {par}: {e}")
                    continue
        else:
            for index, par in enumerate(param_values[1:], start=0):
                try:
                    par_array[par_index] = par
                    sol = wrapper(par_array, equilibria[index])
                    equilibria[index+1] = sol
                except Exception as e:
                    warnings.warn(f"Failed to find limit cycle at parameter {par}: {e}")
                    continue
        return param_values, equilibria
    except Exception as e:
        warnings.warn(f"An error occurred in natural continuation: {e}")
        return None, None

def pseudo_continuation(ode, x0, par_array, par_index, min_par, max_par, max_steps, wrapper, phase_condition, increase):

    """
    Performs pseudo-arclength continuation on a system of ODEs to track solution branches as parameters vary.

    Args:
        ode (callable): The ordinary differential equation system to solve.
        x0 (np.ndarray): Initial guess for the solution.
        par_array (list): Array of parameters used in the ODE.
        par_index (int): Index of the parameter in par_array that will be varied.
        min_par (float): Minimum value of the continuation parameter.
        max_par (float): Maximum value of the continuation parameter.
        max_steps (int): Maximum number of continuation steps.
        wrapper (callable): A wrapper function to apply the corrector method.
        phase_condition (callable): A function to enforce the phase condition.
        increase (bool): Direction of continuation; True for increasing, False for decreasing.

    Returns:
        tuple: (alpha_sol, u_sol) where alpha_sol are the continuation parameters and u_sol are the solutions.
    """

    try:
        #par_step = np.sign(max_par - min_par) * (max_par - min_par) / max_steps if increase else -(max_par - min_par) / max_steps
        par_step = abs((max_par - min_par) / max_steps) * (1 if increase else -1)
        u_sol, alpha_sol = [], []

        # Prepare initial parameters
        initial_params = par_array.copy()
        initial_params[par_index] = min_par if increase else max_par
        initial_solutions = find_initial_sols(ode, x0, phase_condition, par_index, initial_params, par_step, wrapper)

        # Check for valid initial solutions
        if initial_solutions[0] is None or initial_solutions[1] is None:
            raise ValueError("Initial solutions could not be found, terminating the continuation process.")

        u1, u2 = initial_solutions
        u_sol.append(u1[:-1])  # append the first solution minus the parameter
        alpha_sol.append(u1[-1])  # append the first parameter

        # Initial setup for the while loop
        current_u = u2
        current_param = u2[-1]  # Last element in the solution should be the parameter if structured correctly

        while  min_par <= current_param <= max_par:
            u_pred, delta_u = predict(current_u, u1)
            updated_param = u_pred[-1]  # updated parameter value after prediction

            correction_result = root(lambda u: corrector(ode, u, par_index, par_array, u_pred, delta_u, current_u, par_step, phase_condition), u_pred, method='lm', tol=1e-6)
            if not correction_result.success:
                warnings.warn(f"Correction step failed at parameter {updated_param}: {correction_result.message}")
                break

            u_corrected = correction_result.x
            if u_corrected[-1] < min_par or u_corrected[-1] > max_par:
                print("Parameter boundary reached or exceeded.")
                break

            u_sol.append(u_corrected[:-1])
            alpha_sol.append(u_corrected[-1])
            u1, current_u = current_u, u_corrected  # Update the previous and current solutions for next iteration
        return alpha_sol, u_sol
    except Exception as e:
        warnings.warn(f"An error occurred in pseudo continuation: {e}")
        return None, None

def predict(u_current, u_previous):
    """
    Predicts the next solution estimate based on the secant method.

    Args:
        u_current (np.ndarray): Current solution estimate.
        u_previous (np.ndarray): Previous solution estimate.

    Returns:
        tuple: (u_pred, delta_u) where u_pred is the predicted next solution and delta_u is the difference vector used for the prediction.
    """
    try:
        delta_u = u_current - u_previous  # secant
        u_pred = u_current + delta_u
        return u_pred, delta_u
    except Exception as e:
        warnings.warn(f"Prediction step failed: {e}")
        return None, None


def corrector(ode, u, par_index, par_array, u_pred, delta_u, u_old, par_step, phase_condition):
    """
    Corrector step in the pseudo-arclength continuation, adjusting the predicted solution to satisfy the ODE and the continuation constraint.

    Args:
        ode (callable): The ODE function.
        u (np.ndarray): Current guess for the solution.
        par_index (int): Index of the parameter in the parameter array that is being continued.
        par_array (list): Array of parameters used in the ODE.
        u_pred (np.ndarray): Predicted solution from the previous step.
        delta_u (np.ndarray): Difference vector from the previous prediction step.
        u_old (np.ndarray): Solution from the previous continuation step.
        par_step (float): Step size for the parameter being continued.
        phase_condition (callable): Function to enforce the phase condition.

    Returns:
        np.ndarray: Corrected solution that satisfies the ODE and the continuation constraint.
    """
    try:
        # Calculate the secant vector components for u and parameter
        secant_u = delta_u[:-1]  
        secant_p = delta_u[-1]   

        # Calculate the pseudo-arclength constraint
        pAL = np.dot(u[:-1] - u_pred[:-1], secant_u) + np.dot(u[-1] - u_pred[-1], secant_p)

        # Update parameter array for the new computation
        par_array[par_index] = u[-1]

        # Distinguish between systems with and without phase conditions
        if phase_condition is None:
            # For systems without a phase condition, calculate the ODE residuals directly
            try:
                ode_res = ode(0, u[:-1], par_array)  
                return np.append(ode_res, pAL)
            except Exception as e:
                warnings.warn(f"ODE computation failed: {e}")
                return np.full_like(u, np.inf)  # Return an array of inf to indicate failure in root finding
        else:
            # For systems with a phase condition, use the shooting method or other appropriate method
            try:
                G = shoot(ode, phase_condition)  
                shoot_res = G(u[:-2], u[-2], par_array)
                return np.append(shoot_res, pAL)
            except Exception as e:
                warnings.warn(f"Shooting method computation failed: {e}")
                return None 
    except Exception as e:
        warnings.warn(f"An error occurred in the corrector function: {e}")
        return None 

def find_initial_sols(f, u0_guess, phase_condition, par_index, par0, par_step, wrapper):
    """
    Finds initial solutions for starting the continuation process.

    Args:
        f (callable): The function defining the system of equations.
        u0_guess (np.ndarray): Initial guess for the solutions.
        phase_condition (callable): Function defining the phase condition.
        par_index (int): Index of the parameter in the parameter array.
        par0 (list): Initial parameter values.
        par_step (float): Step size to be used for the continuation parameter.
        wrapper (callable): A wrapper function that applies the discretization and solving methods.

    Returns:
        list: List of initial solutions, each as an np.ndarray.
    """
    try:
        # Initialize solutions array
        solutions = []

        # Adjust the parameter for the first solution
        par0[par_index] += 0  # No change for the first solution
        if phase_condition is None:
            # Find the first solution without phase conditions
            u1 = fsolve(lambda u: f(0, u, par0), u0_guess)
            u1 = np.append(u1, par0[par_index])
        else:
            # Use the wrapper for phase-conditioned systems
            u1 = wrapper(par0, u0_guess)
            if u1 is None:
                warnings.warn(f"Wrapper failed to find initial solution for parameters {par0}")
            u1 = np.append(u1, par0[par_index])

        solutions.append(u1)

        # Adjust the parameter for the second solution
        par0[par_index] += par_step
        if phase_condition is None:
            u2 = fsolve(lambda u: f(0, u, par0), u1[:-1])
            u2 = np.append(u2, par0[par_index])
        else:
            u2 = wrapper(par0, u1[:-1])
            if u2 is None:
                warnings.warn(f"Wrapper failed to find second solution for updated parameters {par0}")
            u2 = np.append(u2, par0[par_index])

        solutions.append(u2)

        return solutions
    except Exception as e:
        warnings.warn(f"An error occurred in finding initial solutions: {e}")
        return [None, None]


def natural_plotter(param_values, solutions, period=True):
    plt.figure(figsize=(12, 6))
    # Check if solutions is 1D or 2D
    if period == True:
        t = 1
    else:
        t = 0

    if solutions.ndim == 1:
        # If 1D, plot directly
        plt.plot(param_values, solutions, label='Solution')
    else:
        # If 2D, plot as before
        for i in range(solutions.shape[1]-t):# Exclude the last component if it's time or similar
            plt.plot(param_values, solutions[:, i], label=f'u{i+1}')
    
    plt.xlabel('Parameter', fontsize=12)
    plt.ylabel('Solution Components', fontsize=12)
    plt.title('Natural Continuation Bifurcation Diagram', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

def pseudo_plotter(par, sol, period=False):
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
    plt.plot(par, norms, label='Norm of the solutions')
    # Plot each component
    if period == True:
        components = components[:-1] #If theres a period in components, exclude it  
    else:
        components = components #If there isnt a period in components, keep the same
    
    for i, component_data in enumerate(components[:]):  # Exclude the last component if it's time or similar
        plt.plot(par, component_data, label=f'u{i+1}') #, marker='.', linestyle='--')
    
    plt.xlabel('Parameter', fontsize=12)
    plt.ylabel('Solution Magnitude', fontsize=12)
    plt.title('Pseudo Arc Length Method', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()




def main():


    def phase_condition(ode,u0,pars):
    #return the phase condition which is du1/dt(0) = 0
        return np.array([ode(0,u0,pars)[0]])
    
    def cubic(t,x, param):
        f = x**3 - x - param
        return f
    
    x0 = np.array([0.37, 3.5, 7.15])
    # x0 = np.array([1]) # Continuation is all dependent on the initial condition. Ensure this is right for cubic
    par_array = [3,1]  # Start parameter value (B & A)
    par_index = 0 # Index of the parameter to continue
    max_steps = [200, 70]
    phase_condition = phase_condition

    # par_pseudo, sol_pseudo = numerical_continuation(cubic, 'pseudo', x0, par_array, par_index, [3, 1.5], [200, 70], shoot, fsolve, phase_condition=phase_condition, increase=False)

    #par_nat, sol_nat = numerical_continuation(brusselator, 'natural', x0, par_array, par_index, [3, 1.5], [200, 70], shoot, fsolve, phase_condition=phase_condition, increase=False)
    
    # natural_plotter(par_nat[1:], sol_nat[1:], period=True)
    # pseudo_plotter(par_pseudo, sol_pseudo, period=False)

    par_pseudo, sol_pseudo = numerical_continuation(brusselator, 
    'pseudo', 
    x0, 
    par_array, 
    par_index, 
    [3, 1.5],
    [200, 30], 
    shoot, 
    fsolve, 
    phase_condition=phase_condition, 
    increase=False)
    # x0 = [1.04, 0.52, -0.52, 3.63] #starting point from b) 
    # par_array = [1]  # parameter value - becomes redundant anyway as we start from max unless there are more parameters
    # par_index = 0

    # par_pseudo, sol_pseudo = numerical_continuation(hopf_bifurcation_3d, 
    #     'pseudo', 
    #     x0, 
    #     par_array, 
    #     par_index, 
    #     [1, -0.6], #Bounds affect the step size
    #     [200, 30], #Max steps: [PAL, Natural]. Also affects the step size  
    #     shoot, 
    #     fsolve, 
    #     phase_condition=phase_condition, 
    #     increase=False)

    # pseudo_plotter(par_pseudo, sol_pseudo, period=True) #using solve_ivp instead of solve_ode somehow makes PAL miss the limit cycle - probably due to adaptive step size

if __name__ == "__main__":
   main()

#TODO: Takes a while to run for more complex systems. Need to optimize the code.
