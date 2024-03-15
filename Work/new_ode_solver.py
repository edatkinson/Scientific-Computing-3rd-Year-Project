import numpy as np 
import matplotlib.pyplot as plt 
import sys
import math
from Equations_Functions import system_of_odes, dx_dt, lokta_volterra
import warnings

def euler_step(f,x0,t0,h, **kwargs):
    x1 = x0 + h*f(t0,x0,**kwargs)
    t1 = t0 + h
    return x1, t1

def rk4_step(f,x0,t0,h, **kwargs):
    k1 = f(t0,x0,**kwargs)
    # print(x0.shape)
    k2 = f(t0+h/2, x0+h*(k1/2),**kwargs)
    k3 = f(t0+h/2,x0+h*(k2/2),**kwargs)
    k4 = f(t0+h,x0+h*k3,**kwargs)
    x1 = x0 + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    t1 = t0 + h
    return x1, t1


def solve_ode(f, t_span, y0, h,method='rk4', **kwargs):
    """
    A custom ODE solver for n-dimensional systems that mimics solve_ivp's Sol.y.
    
    Parameters:
    - f: The ODE system function.
    - t_span: Tuple of (start, end) times.
    - y0: Initial conditions (n-dimensional).
    - method: Numerical method ('Euler' or 'RK4').
    - h: Step size.
    - **kwargs: Additional arguments for the ODE system function.
    
    Returns:
    - t: Array of time points.
    - y: Solution array (n-dimensional).
    """

    t0, t_end = t_span

    num_steps = int(abs(np.ceil((t_end - t0) / h)))

    t = np.linspace(t0, t_end, num_steps+1)
    y = np.zeros((len(y0), len(t)))
    y[:, 0] = y0

    # print(num_steps)
    # print(t_end)
    for i in range(1, len(t)):

        dt = min(h, t_end - t[i-1])
        #print(dt)

        if method == 'euler':
            y[:, i], _ = euler_step(f, y[:, i-1], t[i-1], dt, **kwargs)
        elif method == 'rk4':
            y[:, i], _ = rk4_step(f, y[:, i-1], t[i-1], dt, **kwargs)

        
    return t, y.T  # Transpose y to match Sol.y structure


# Solve the system


def main():

    pars = (1,0.1,0.1) #Vary these to see changes in the plot

    t, x = solve_ode(lokta_volterra, (0,100), np.array([1,2]),method='rk4',h=0.01,pars=pars)
    print(x)
    plt.plot(t, x[:, 0], label='Prey')
    plt.plot(t, x[:, 1], label='Predator')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
