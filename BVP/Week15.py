
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
from scipy.optimize import fsolve

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


beta = 0.1  # or any other value beta is between 0.1 and 0.5
t = np.linspace(0, 100, 500)  # time grid
x0 = [0.1, 0.1]  # initial conditions

sol = odeint(lokta_volterra, x0, t, args=(beta,)) #using the odeint from scipy lib

#Find the roots
roots = fsolve(lokta_volterra, x0, args=(beta,))
print('Roots:', roots)

plt.plot(t, sol[:, 0], label='Prey')
plt.plot(t, sol[:, 1], label='Predator')
plt.legend()
#plt.savefig('Beta = 0.1.png')
plt.show()





