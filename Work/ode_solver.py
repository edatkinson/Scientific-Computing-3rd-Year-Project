import numpy as np 
import matplotlib.pyplot as plt 
import sys
import math
from Equations_Functions import system_of_odes, dx_dt, lokta_volterra, brusselator, hopf_bifurcation_3d, hopf_3dim
import warnings


def euler_step(f, x0, t0, h, *args):
    
    x1 = x0 + h * f(t0, x0, *args)
    t1 = t0 + h
    return x1, t1


def rk4_step(f, x0, t0, h, *args):
    
    k1 = np.array(f(t0, x0, *args))
    k2 = np.array(f(t0 + h / 2, x0 + (h / 2) * k1, *args))
    k3 = np.array(f(t0 + h / 2, x0 + (h / 2) * k2, *args))
    k4 = np.array(f(t0 + h, x0 + h * k3, *args))
    x1 = x0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    t1 = t0 + h
    return x1, t1


def solve_to(step, f, x1, t1, t2, deltat_max, *args):

    while (t1 + deltat_max) < t2:  
        x1, t1 = step(f, x1, t1, deltat_max, *args)
    else:
        x1, t1 = step(f, x1, t1, t2 - t1, *args)  
    
    return x1

def solve_ode(f, x0, t, method, deltat_max, *args):
    
    # Choose the stepping method based on the method argument
    if method == "euler":
        step = euler_step
    elif method == "rk4":
        step = rk4_step
    else:
        raise ValueError(f"Method is not recognised: {method}. Choose 'euler' or 'rk4'.")

    # Initialise sol array
    sol= np.zeros((len(t), len(x0)))
    sol[0] = x0

    # Solve f at each time step
    for i in range(1, len(t)):
        t1, t2 = t[i-1], t[i]
        deltat = min(deltat_max, t2 - t1)
        x1 = sol[i-1]
        
        # Solve from t1 to t2 with the chosen method
        x, _ = step(f, x1, t1, deltat, *args)
        sol[i] = x

    return sol.T

def plot_errors(dx_dt, true_solution, x0, t0, tf):
    timestep_values = np.logspace(-6, -1, 10)  # From 1e-6 to 1e-1
    errors_euler = []
    errors_rk4 = []

    for h in timestep_values:
        # Solve using Euler's method
        x_euler = solve_to(euler_step, dx_dt, x0, t0, tf, h)
        true_val_euler = true_solution(tf)
        error_euler = np.abs(true_val_euler - x_euler)
        errors_euler.append(error_euler)

        # Solve using RK4 method
        x_rk4 = solve_to(rk4_step, dx_dt, x0, t0, tf, h)
        true_val_rk4 = true_solution(tf)
        error_rk4 = np.abs(true_val_rk4 - x_rk4)
        errors_rk4.append(error_rk4)

    # Plotting
    fig, ax = plt.subplots()
    ax.loglog(timestep_values, errors_euler, label='Euler', marker='o', color='red')
    ax.loglog(timestep_values, errors_rk4, label='RK4', marker='o')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Error')
    ax.legend()
    plt.show()

    return fig



def main():

    def system_of_odes(t,x):
        dxdt = x[1]
        dydt = -x[0]
        return np.array([dxdt, dydt])
    
    def dx_dt(t,x):
        dx_dt = x
        return dx_dt

    def true_solution(t):
        return np.exp(t)
    
    x0 = np.array([1,1])
    pars = []
    T = 20
    t = np.linspace(0, T, 1000)
    sol = solve_ode(system_of_odes, x0, t, "rk4", 0.05)

    plt.plot(t, sol[0, :], label='x')
    plt.xlabel('Time')
    plt.ylabel('2D ODE')
    plt.legend()
    plt.show()

    #Plot Errors
    y0 = np.array([1])
    fig2 = plot_errors(dx_dt, true_solution, y0, t0=0, tf=2)
    fig2.show()

if __name__ == "__main__":
    main()
