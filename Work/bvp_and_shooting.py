
import numpy as np  
import matplotlib.pyplot as plt 
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
def lokta_volterra(t,x,beta):
    #x = [x,y]
    alpha = 1
    delta = 0.1
    dxdt = x[0]*(1-x[0]) - (alpha*x[0]*x[1])/(delta + x[0])
    dydt = beta * x[1] * (1 - (x[1]/x[0]))
    return np.array([dxdt,dydt])

def equations_to_solve(x, beta):
    alpha = 1
    delta = 0.1
    dxdt = x[0]*(1 - x[0]) - (alpha * x[0] * x[1]) / (delta + x[0])
    dydt = beta * x[1] * (1 - (x[1] / x[0]))
    return [dxdt, dydt]
beta = 0.26
#initial conditions: 
x0 = [0.27015621,0.3]
tspan = [0,150]
sol = solve_ivp(lokta_volterra, tspan, x0, args=(beta,), t_eval=np.linspace(*tspan,10000))
#sol has solutions sol.t and sol.y where sol.y = [x,y]

roots = fsolve(equations_to_solve, x0, args=(beta,))
print(f"Roots: {roots}")

dxdt = np.gradient(sol.y[0], sol.t)
dydt = np.gradient(sol.y[1], sol.t)



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(sol.t, sol.y[0], label='Prey')
plt.plot(sol.t, sol.y[1], label='Predator')
plt.annotate('Initial conditions: ({:.2f}, {:.2f})'.format(*x0), xy=(0.5, 0.1), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=2))
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()

# Phase space plot
plt.subplot(1, 2, 2)
plt.plot(sol.y[0], sol.y[1],alpha=0.5)
plt.xlabel('Prey Population')
plt.ylabel('Predator Population')
plt.title('Phase Space')


arrows = np.arange(0,len(sol.t),100)
for i in arrows:
    plt.quiver(sol.y[0][i],sol.y[1][i],dxdt[i],dydt[i],angles='xy',scale_units='xy',scale=1.6, alpha=0.3)

x = np.linspace(0.1, 0.5, 30)
y = np.linspace(0.1, 0.5, 30)
X, Y = np.meshgrid(x, y)
plt.xlim(0.18,0.4)
plt.ylim(0.18,0.35)

# Compute the rate of change at each point on the grid
U, V = lokta_volterra(0, [X, Y], beta)

# Normalize the arrows so their size represents the speed
N = np.sqrt(U**2 + V**2)
U /= N
V /= N
plt.quiver(X, Y, U, V, angles='xy')
plt.tight_layout()
plt.savefig('Periodic_orbit.png')
plt.show()



