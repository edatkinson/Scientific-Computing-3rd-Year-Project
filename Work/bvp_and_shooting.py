
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve, root
from scipy.integrate import solve_ivp
from new_ode_solver import solve_ode
from new_ode_solver import solve_ode

from Equations_Functions import lokta_volterra, hopf, hopf_3dim, modified_hopf

#Ignore runTime warnings
import warnings
from scipy.optimize import OptimizeWarning
warnings.filterwarnings('ignore', category=OptimizeWarning)

#Shoots using the scipy solve_ivp function

'''
To find limit cycles, we must solve the periodic boundary value problem (BVP)
Solve u(0) - u(T) = 0, = u0 - F(u0, T) = 0.

T = 2pi/omega

Hence, limit cycles of (3) can be found by passing (6) along with a suitable initial guess u ̃ 0 to a numerical root finder such as fsolve in Matlab or Python (Scipy) or nlsolve in Julia.
All of the above can be trivially generalised to arbitrary periodically- forced ODEs of any number of dimensions.

'''


#####Root finding problem and shooting########

def phase_condition(ode,u0,pars):
    #return the phase condition which is du1/dt(0) = 0
    return np.array([ode(0,u0,pars)[0]])

def shoot(f, phase_cond=None):

    #Check if there is a phase condition
    if phase_cond is None:
        def phase_cond_noop(f, u0, pars):
            return np.array([])
        phase_cond = phase_cond_noop
    elif not callable(phase_cond):
        raise TypeError("The phase_condition argument must be a callable function or None.")


    def G(u0, T, pars):
    
    
        # Solve ODE system using Solve IVP 
        t = np.linspace(0, T, 1000)
        try:
            # sol = solve_ivp(f, (0, T), u0, t_eval=t,args=(pars,))
            # final_sol = sol.y[:,-1]
            
            # _ , sol = solve_ode(f, (0,T), u0, h=0.001,method='euler',pars=pars)
            # final_sol = sol[-1, :]
            t = np.linspace(0, T, 1000)
            sol = solve_ode(f, u0, t, "rk4", 0.05, pars)
            final_sol = sol[:, -1]
            #print(final_sol)
            if np.isnan(sol).any():
                raise ValueError("The ODE solver returned NaN values, which indicates a problem with the ODE integration.")
            return np.append(u0 - final_sol, phase_cond(f, u0, pars=pars))
        except Exception as e:
            raise RuntimeError(f"An error occurred during shooting: {e}")

    return G


def orbit(ode, uinitial, duration,pars):
    t = np.linspace(0,duration,150)
    sol = solve_ivp(ode, (0, duration), uinitial,t_eval=t ,args=(pars,))
    
    return sol.y

def limit_cycle_finder(ode, estimate, phase_condition, pars, test=False):
    G = shoot(ode,phase_condition)
    try:
        solution, info, ier, msg = fsolve(lambda estimate: G(estimate[:-1], estimate[-1], pars), estimate, full_output=True)
        if ier != 1:
            raise OptimizeWarning(f"Root finder failed to converge: {msg}")
        if test:
            print("Root finder convergence: PASSED")
            print(f"Root finder solution: {solution}")
        return np.array(solution), orbit(ode, solution[:-1], solution[-1], pars=pars)

    except OptimizeWarning as e:
        warnings.warn(str(e), OptimizeWarning)
        return solution, orbit(ode, solution[:-1], solution[-1], pars=pars) 
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")



def phase_portrait_plotter(sol):
    plt.plot(sol[0,:], sol[1,:], label='Isolated periodic orbit')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Phase portrait')
    plt.legend()

    #Code to plot the 3D Hopf limit cycle:
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(sol[:,0], sol[:,1], sol[:,2], label='Isolated periodic orbit')
    # ax.set_xlabel('u1')
    # ax.set_ylabel('u2')
    # ax.set_zlabel('u3')
    
    return plt#, fig


#Create a Main function which does the code below


def main():
    lokta_pars = (1,0.1,0.1)
    orbit, cycle1 = limit_cycle_finder(lokta_volterra, [0.1,0.1,30],phase_condition,lokta_pars)
    print('The true values of the Lokta-Volterra orbit:', orbit)
    fig1 = phase_portrait_plotter(cycle1) #plot the limit cycle
    plt.show()

    hopf_pars = (0.9,-1)
    orbit, cycle2 = limit_cycle_finder(hopf, [2,1,5],phase_condition,hopf_pars)
    print('The true values of the Hopf orbit:', orbit)
    fig2 = phase_portrait_plotter(cycle2) #plot the limit cycle
    plt.show()

    t = np.linspace(0,10,100)
    beta,sigma = hopf_pars
    theta = 1
    u1 = beta**0.5 * np.cos(t+theta)
    u2 = beta**0.5 * np.sin(t+theta)
    plt.plot(u1,u2) #plot the analytical phase portrait of the limit cycle
    plt.show()

if __name__ == "__main__":
    main()




    
