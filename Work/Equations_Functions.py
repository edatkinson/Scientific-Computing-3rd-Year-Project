import numpy as np


def system_of_odes(t,x):
    dxdt = x[1]
    dydt = -x[0]
    return np.array([dxdt, dydt])

def dx_dt(t,x):
    dx_dt = x
    return dx_dt

def lokta_volterra(t,x,pars):
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
    beta, sigma = params
    du1dt = beta*u[0] - u[1] + sigma*u[0] * ((u[0])**2 + (u[1])**2)
    du2dt = u[0] + beta*u[1] + sigma*u[1] * ((u[0])**2 + (u[1])**2)
    du3dt = -u[2]
    dUdt = np.array([du1dt,du2dt,du3dt])
    return dUdt

def modified_hopf(t,u, pars):
    """
    Returns the time derivative of a 2D predator-prey system at a given time.

    :param U: A numpy array of length 2 containing the state variables (u1, u2).
    :param t: The time.
    :param pars: A tuple containing the system parameters (beta).

    :returns: A numpy array of length 2 containing the time derivatives of the state variables.
    """
    (u1, u2) = u
    du1dt = (pars * u1) - u2 + u1 * (u1**2 + u2**2) - (u1 * (u1**2 + u2**2)**2)
    du2dt = u1 + (pars * u2) + u2 * (u1**2 + u2 ** 2)- (u2 * (u1**2 + u2 ** 2)**2)
    return np.array([du1dt, du2dt])