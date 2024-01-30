
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

#to Isolate a periodic orbit, 


def lokta_volterra(x,t,beta):
    alpha = 1
    delta = 0.1
    dxdt = x[0]*(1-x[0]) - (alpha*x[0]*x[1])/(delta + x[0])
    dydt = beta * x[1] * (1 - (x[1]/x[0]))
    return np.array([dxdt,dydt])


beta = 0.1  # or any other value beta is between 0.1 and 0.5
t = np.linspace(0, 200, 500)  # time grid
x0 = [0.1, 0.1]  # initial conditions

sol = odeint(lokta_volterra, x0, t, args=(beta,)) #using the odeint from scipy lib

plt.plot(t, sol[:, 0], label='Prey')
plt.plot(t, sol[:, 1], label='Predator')
plt.legend()
#plt.savefig('Beta = 0.1.png')
plt.show()





