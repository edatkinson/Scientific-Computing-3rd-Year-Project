
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
#to Isolate a periodic orbit, 
'''
To find limit cycles, we must solve the periodic boundary value problem (BVP)
Solve u(0) - u(T) = 0, = u0 - F(u0, T) = 0.

T = 2pi/omega

Hence, limit cycles of (3) can be found by passing (6) along with a suitable initial guess u ̃ 0 to a numerical root finder such as fsolve in Matlab or Python (Scipy) or nlsolve in Julia.
All of the above can be trivially generalised to arbitrary periodically- forced ODEs of any number of dimensions.

'''
def lokta_volterra(x,beta,t=None):
    alpha = 1
    delta = 0.1
    dxdt = x[0]*(1-x[0]) - (alpha*x[0]*x[1])/(delta + x[0])
    dydt = beta * x[1] * (1 - (x[1]/x[0]))
    return np.array([dxdt,dydt])


beta_values = np.linspace(0.1, 0.5, 40)  # Explore beta in [0.1, 0.5]
t = np.linspace(0, 100, 10000)  # Time grid

# Initial condition range
x0_range = [0.1, 0.5]
y0_range = [0.1, 0.5]
initial_conditions = np.linspace(x0_range[0], y0_range[1], 10)

# Iterate over beta values and initial conditions
for beta in beta_values:
    plt.figure(figsize=(10, 8))
    for x0 in initial_conditions:
        for y0 in initial_conditions:
            sol = odeint(lokta_volterra, [x0, y0], t, args=(beta,))
            plt.plot(sol[:, 0], sol[:, 1], label=f'x0={x0:.2f}, y0={y0:.2f}')
    
    plt.xlabel('Prey')
    plt.ylabel('Predator')
    plt.title(f'Phase Portrait for beta={beta:.2f}')
    plt.legend()
    plt.show()


