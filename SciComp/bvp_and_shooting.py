
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve, root
from scipy.integrate import solve_ivp
from ode_solver import solve_ode

from Equations_Functions import lokta_volterra, hopf, hopf_3dim, modified_hopf

#Ignore runTime warnings
import warnings
from scipy.optimize import OptimizeWarning
warnings.filterwarnings('ignore', category=OptimizeWarning)


#####Root finding problem and shooting########

def phase_condition(ode,u0,pars):
    """
    Calculate the phase condition for a system of ordinary differential equations (ODEs).
    
    This function evaluates the phase condition, which is the derivative of the first state variable
    with respect to time at t = 0, and it should be zero to satisfy the phase condition.

    Args:
        ode (callable): A function representing the system of ODEs. It must accept three parameters:
                        time (scalar), state vector (array), and parameters (tuple).
        u0 (array): Initial condition of the system.
        pars (tuple or list): Parameters required for the ODE function.

    Returns:
        array: The evaluated phase condition, specifically the value of the first derivative of the
               first state variable at t = 0.
    """
    return np.array([ode(0,u0,pars)[0]])

def shoot(f, phase_cond):
    """
    Generate a boundary-value problem solver using the shooting method.
    
    This function creates a solver for finding periodic solutions in a system of ODEs by adjusting
    the initial conditions to match the final conditions after one period.

    Args:
        f (callable): The function representing the system of ODEs.
        phase_cond (callable): The function to evaluate the phase condition.

    Returns:
        callable: A function G that accepts initial conditions, period T, and parameters, and
                  calculates the discrepancy between the desired and actual final states, including
                  the phase condition.
    """
    def G(u0, T, pars):

        t = np.linspace(0, T, 1000)
        try:
            sol = solve_ode(f, u0, t, "rk4", 0.01, pars) #solve the the ode from 0 to T 
            final_sol = sol[:, -1] #Excracr the final solution
            if np.isnan(sol).any():
                raise ValueError("The ODE solver returned NaN values, which indicates a problem with the ODE integration. Ensure you are using the correct Initial Conditions.") #Check for NaN values and return error if found
            if phase_cond == None:
                return 0
            else:
                return np.append(u0 - final_sol, phase_cond(f, u0, pars))
        except Exception as e:
            raise RuntimeError(f"An error occurred during shooting: {e}")

    return G


def orbit(ode, uinitial, duration,pars):
    """
    Solve an ODE system over a specified duration to generate an orbit.
    
    Args:
        ode (callable): The system of ODEs to be solved.
        uinitial (array): Initial conditions for the ODE solver.
        duration (float): Time interval over which to solve the ODE.
        pars (tuple): Parameters required for the ODE function.

    Returns:
        tuple: A tuple containing the solution array and the time array.
    """
    t = np.linspace(0,duration,150)
    sol = solve_ivp(ode, (0, duration), uinitial,t_eval=t ,args=(pars,))
    
    return sol.y, t

def limit_cycle_finder(ode, estimate, phase_condition, pars,descretization=shoot, solver=fsolve, test=False):
    """
    Find a limit cycle for a system of ODEs using a specified root-finding method and shooting method.

    Args:
        ode (callable): The system of ODEs.
        estimate (array): Initial guess for the root-finder, including initial conditions and period.
        phase_condition (callable): Function to evaluate the phase condition.
        pars (tuple): Parameters for the ODEs.
        descretization (callable, optional): Function to convert the boundary value problem to an
                                             initial value problem via the shooting method.
        solver (callable, optional): Root-finding method to solve the initial value problem.
        test (bool, optional): If True, print additional diagnostics about the solver.

    Returns:
        array: The solution array containing the final initial conditions and period that approximately
               satisfy the boundary conditions, forming a limit cycle.
    """
    G = descretization(ode,phase_condition)
    try:
        solution, info, ier, msg = solver(lambda estimate: G(estimate[:-1], estimate[-1], pars), estimate, full_output=True)
        if ier != 1:
            raise OptimizeWarning(f"Root finder failed to converge: {msg}")
        if test:
            print("Root finder convergence: PASSED")
            print(f"Root finder solution: {solution}")
        return np.array(solution)

    except OptimizeWarning as e:
        warnings.warn(str(e), OptimizeWarning)
        return solution 
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


#Create a Main function which does the code below


def main():
    lokta_pars = (1,0.1,0.1)
    sol = limit_cycle_finder(lokta_volterra, [0.1,0.2,30],phase_condition,lokta_pars, descretization=shoot, solver=fsolve)
    cycle1 = orbit(lokta_volterra, sol[:-1], sol[-1], lokta_pars)
    print('The true values of the Lokta-Volterra orbit:', sol)


    hopf_pars = (0.9,-1)
    sol = limit_cycle_finder(hopf, [2,1,5],phase_condition,hopf_pars, descretization=shoot, solver=fsolve)
    cycle2 = orbit(hopf, sol[:-1], sol[-1], hopf_pars)
    print('The true values of the Hopf orbit:', sol)

    t = np.linspace(0,10,100)
    beta,sigma = hopf_pars
    theta = 1
    u1 = beta**0.5 * np.cos(t+theta)
    u2 = beta**0.5 * np.sin(t+theta)
    plt.plot(u1,u2) #plot the analytical phase portrait of the limit cycle
    plt.show()

if __name__ == "__main__":
    main()



