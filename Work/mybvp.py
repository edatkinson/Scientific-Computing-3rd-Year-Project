
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve, root
from scipy.integrate import solve_ivp
from new_ode_solver import solve_ode

import warnings
from Equations_Functions import lokta_volterra, hopf, hopf_3dim, modified_hopf
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

    def J(u0,T,pars):
        # Solve ODE system using Solve IVP 
        #t = np.linspace(0, T, 100)
        _,sol = solve_ode(f, (0,T), u0, h=0.01,method='rk4',pars=pars)
        final_sol = sol[-1, :]
        #print(final_sol)
        # Calculate differences between actual and estimates
        return np.append(u0 - final_sol, phase_cond(f, u0, pars=pars))

    return J


def orbit_my(ode, uinitial, duration,pars):
    _,sol = solve_ode(ode, (0,duration),uinitial, h=0.01,method='rk4',pars=pars)
    #print(sol[:,-1])
    return sol

def limit_cycle_finder(ode, estimate, phase_condition, pars):
    J = shoot(ode,phase_condition)
    #root finding problem
    result_my = fsolve(lambda estimate: J(estimate[:-1], estimate[-1], pars),estimate, xtol=1e-6, epsfcn=1e-6)

    myisolated_orbit = orbit_my(ode, result_my[0:-1],result_my[-1],pars=pars) #result[-1] is the period of the orbit, result[0:-1] are the initial conditions
    myisolated_orbit = myisolated_orbit
    
    #Isolated_Orbit is the numerical approximation of the limit cycle ODE

    return result_my, myisolated_orbit

def phase_portrait_plotter(sol):
    plt.plot(sol[:,0], sol[:,1], label='Isolated periodic orbit')
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
    orbit, cycle2 = limit_cycle_finder(hopf, [1,0.1,7],phase_condition,hopf_pars)
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

