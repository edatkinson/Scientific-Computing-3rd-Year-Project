
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from odesolver import solve_ode

'''
To find limit cycles, we must solve the periodic boundary value problem (BVP)
Solve u(0) - u(T) = 0, = u0 - F(u0, T) = 0.

T = 2pi/omega

Hence, limit cycles of (3) can be found by passing (6) along with a suitable initial guess u ̃ 0 to a numerical root finder such as fsolve in Matlab or Python (Scipy) or nlsolve in Julia.
All of the above can be trivially generalised to arbitrary periodically- forced ODEs of any number of dimensions.

'''


def lokta_volterra(t,x,params: list):
    alpha, delta, beta = params
    dxdt = x[0]*(1-x[0]) - (alpha*x[0]*x[1])/(delta + x[0])
    dydt = beta * x[1] * (1 - (x[1]/x[0]))
    dXdt = np.array([dxdt,dydt])
    return dXdt

def hopf(t,u,params: list):#params = [beta, sigma]
    beta, sigma = params
    du1dt = beta*u[0] - u[1] + sigma*u[0] * ((u[0])**2 + (u[1])**2)
    du2dt = u[0] + beta*u[1] + sigma*u[1] * ((u[0])**2 + (u[1])**2) 
    dUdt = np.array([du1dt,du2dt])
    return dUdt

def hopf_3dim(t,u,params: list):
    beta, sigma = params
    du1dt = beta*u[0] - u[1] + sigma*u[0] * ((u[0])**2 + (u[1])**2)
    du2dt = u[0] + beta*u[1] + sigma*u[1] * ((u[0])**2 + (u[1])**2)
    du3dt = -u[2]
    dUdt = np.array([du1dt,du2dt,du3dt])
    return dUdt


#####Root finding problem and shooting########

def ode(function: callable, params: list): #callable function, params = list of parameters
    return lambda t, U: function(t, U, params) #returns a function of t and U f(t,U) as in the notes


#integrator using my ode_solve function for the shooting problem
def integrate(ode,u0,T): #u0 is the initial guess [u1, u2]
    t = np.linspace(0,T,150)
    #solv = solve_ivp(ode,[0,T],u0)
    sol, t = solve_ode(ode,u0,t,'rk4',0.01)
    # x_sol = sol[:, 0] #Extract the x values
    # y_sol = sol[:, 1] #Extract the y values
    #t_sol = t #Extract the time values (array of times)
    #return solv.y[:,-1]
    return sol[-1,:]

def phase_condition(ode,u0,T):
    #return the phase condition which is du1/dt(0) = 0
    return np.array([ode(0,u0)[0]]) 

def shoot(ode, estimate, phase_condition):
    ''' 
    API
    A function which performs numerical shooting.

    Parameters
    __________

    ode : callable function with parameters, f(t,u,params)
    estimate: list of initial guesses for the periodic orbit, e.g. [u1, u2, T]
    phase_condition: callable function, phase_condition(ode,u0,T)
    
    Returns
    _______

    An array of [u0 - A, u1 - B, phasecondition]
    
    '''
    
    u0 = estimate[0:-1] #except the last estimate which is T
    T = estimate[-1] #Estimating the period of the orbit
    return np.hstack((u0-integrate(ode,u0,T),phase_condition(ode,u0,T)))


def orbit(ode, uinitial, duration):
    #sol = solve_ivp(ode, (0, duration), uinitial) 
    t = np.linspace(0,duration,150)
    sol, t = solve_ode(ode,uinitial,t,'rk4',0.01)
    return sol

def limit_cycle_finder(ode, estimate, phase_condition):
    #root finding problem
    result = fsolve(lambda estimate: shoot(ode,estimate,phase_condition),estimate) 
    #Lambda function to pass the function shoot to fsolve to make shoot = 0
    #result = initial conditions of u which makes shoot function = 0
    isolated_orbit = orbit(ode, result[0:-1],result[-1]) #result[-1] is the period of the orbit, result[0:-1] are the initial conditions
    return result, isolated_orbit

def phase_portrait_plotter(sol):
    plt.plot(sol[:,0], sol[:,1], label='Isolated periodic orbit')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Phase portrait')
    plt.legend()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol[:,0], sol[:,1], sol[:,2], label='Isolated periodic orbit')
    ax.set_xlabel('u1')
    ax.set_ylabel('u2')
    ax.set_zlabel('u3')
    
    return plt, fig

#estimate = [u1, u2, T]
initial_guess = [1,1,10] #I get errors when the initial guesses are far away from the limit cycle
initial_guess2 = [1,1,1,100] #[u1, u2, u3, T]
hopf_params = [0.9, -1] #beta = any, sigma = -1
lokta_params = [1,0.1,0.26]  #alpha, delta, beta
roots, limit_cycle = limit_cycle_finder(ode(hopf_3dim,hopf_params),initial_guess2,phase_condition)
print(roots)
fig1,fig2 = phase_portrait_plotter(limit_cycle)
plt.show()

# So [ 0.37355557  0.29663022 36.07224553] are the initial conditions and period of the periodic orbit

# t = np.linspace(0,10,100)

# beta, sigma = hopf_params
# theta = 1
# u1 = beta**0.5 * np.cos(t+theta)
# u2 = beta**0.5 * np.sin(t+theta)

# plt.plot(u1,u2)
# plt.show()

