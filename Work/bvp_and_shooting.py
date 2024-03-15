
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve, root
from scipy.integrate import solve_ivp
from new_ode_solver import solve_ode

import warnings
from Equations_Functions import lokta_volterra, hopf, hopf_3dim, modified_hopf

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

def shoot(f, phase_cond):
    """
    Returns a function G for solving a boundary value problem using the shooting method.

    :param f: System of ODEs function.
    :param phase_cond: Boundary conditions function.
    :returns: Function G calculating differences between actual and guessed boundary conditions.
    """

    def G(u0, T, pars):
        """
        Calculates differences between actual and guessed boundary conditions for a given problem.

        :param u0: Initial guess for the solution.
        :param T: Final time value.
        :param pars: Dictionary of parameter values.
        :returns: Numpy array of differences between actual and guessed boundary conditions.
        """
        # Ignore runTime warnings
        
        # Solve ODE system using Solve IVP 
        t = np.linspace(0, T, 100)
        #sol, t = solve_ode(f, u0, t, method='rk4', h=0.01,pars=pars)
        sol = solve_ivp(f, (0, T), u0, t_eval=t, args=(pars,))
        #final_sol = sol[-1, :]
        final_sol = sol.y[:,-1]
        #print(final_sol)
        # Calculate differences between actual and estimates
        return np.append(u0 - final_sol, phase_cond(f, u0, pars=pars))

    return G

def orbit_ivp(ode, uinitial, duration,pars):
    t = np.linspace(0,duration,150)
    sol = solve_ivp(ode, (0, duration), uinitial,t_eval=t ,args=(pars,))
    #print(sol.y)
    return sol.y

def limit_cycle(ode,estimate,phase_condition,pars):
    J = shoot(ode,phase_condition)
    u0 = estimate[:-1]
    T = estimate[-1]
    pars = pars
    #root finding problem
    result = fsolve(lambda estimate: J(u0,T, pars),estimate, xtol=1e-6, epsfcn=1e-6)
    #Isolated_Orbit is the numerical approximation of the limit cycle ODE

    return result

def limit_cycle_finder(ode, estimate, phase_condition, pars):
    G = shoot(ode,phase_condition)
    #root finding problem
    result_ivp = fsolve(lambda estimate: G(estimate[:-1], estimate[-1], pars),estimate)
  
    ivpisolated_orbit = orbit_ivp(ode, result_ivp[0:-1],result_ivp[-1],pars=pars) #result[-1] is the period of the orbit, result[0:-1] are the initial conditions
    ivpisolated_orbit = ivpisolated_orbit
    
    #Isolated_Orbit is the numerical approximation of the limit cycle ODE

    return result_ivp, ivpisolated_orbit

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




    
