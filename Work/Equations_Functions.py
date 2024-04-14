import numpy as np


def system_of_odes(t,x):
    dxdt = x[1]
    dydt = -x[0]
    return np.array([dxdt, dydt])

def dx_dt(t,x):
    dx_dt = x
    return dx_dt

def lokta_volterra(t,x,pars):

    if not isinstance(pars, (list, tuple, np.ndarray)):
        pars = [pars]
    alpha, delta, beta = pars
    dxdt = x[0]*(1-x[0]) - (alpha*x[0]*x[1])/(delta + x[0])
    dydt = beta * x[1] * (1 - (x[1]/x[0]))
    dXdt = np.array([dxdt,dydt])
    return dXdt

def hopf(t,u,pars):#params = (beta, sigma)
    beta, sigma = pars
    du1dt = beta*u[0] - u[1] + sigma*u[0] * ((u[0])**2 + (u[1])**2)
    du2dt = u[0] + beta*u[1] + sigma*u[1] * ((u[0])**2 + (u[1])**2) 
    dUdt = np.array([du1dt,du2dt])
    return dUdt

def hopf_3dim(t,u,pars):
    beta, sigma = pars
    du1dt = beta*u[0] - u[1] + sigma*u[0] * ((u[0])**2 + (u[1])**2)
    du2dt = u[0] + beta*u[1] + sigma*u[1] * ((u[0])**2 + (u[1])**2)
    du3dt = -u[2]
    dUdt = np.array([du1dt,du2dt,du3dt])
    return dUdt



def modified_hopf(t,u, pars):
  
    beta = pars[0]

    u1, u2 = u
    du1dt = (beta * u1) - u2 + u1 * (u1**2 + u2**2) - (u1 * (u1**2 + u2**2)**2)
    du2dt = u1 + (beta * u2) + u2 * (u1**2 + u2 ** 2)- (u2 * (u1**2 + u2 ** 2)**2)

    return np.array([du1dt, du2dt])


def brusselator(t,u,pars):
    A = 1
    B = pars[0]
    u1,u2 = u
    du1dt = A + u1**2 * u2 - (B + 1) * u1
    du2dt = B * u1 - u1**2 * u2

    dUdt = np.array([du1dt,du2dt])
    return dUdt

def hopf_bifurcation_3d(t, X, pars):
    x, y, z = X
    beta = pars[0]
    r_squared = x**2 + y**2 + z**2  # Common term (r^2) in the equations
    
    # Compute the derivatives
    dxdt = beta*x - y - z + x*r_squared - x*r_squared**2
    dydt = x + beta*y - z + y*r_squared - y*r_squared**2
    dzdt = x + y + beta*z + z*r_squared - z*r_squared**2
    
    return [dxdt, dydt, dzdt]


def hopfNormal(t, u, pars):
    beta = pars[0]
    sigma = 1

    u1, u2 = u

    du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
    du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)

    return [du1dt, du2dt]


def cubic(t,x, param):
    f = x**3 - x - param
    return f

def hopf_example(t,u,pars):#params = (beta)
    beta = pars
    du1dt = beta*u[0] - u[1] - u[0] * ((u[0])**2 + (u[1])**2)
    du2dt = u[0] + beta*u[1] - u[1] * ((u[0])**2 + (u[1])**2) 
    dUdt = np.array([du1dt,du2dt])
    return dUdt