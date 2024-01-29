import numpy as np 
import matplotlib.pyplot as plt 

'''

Euler Notes:
dx/dt = f(t,x) 
where t_new = t_old + h
and x_new = x_old + hf(t_old,x_old)

Runge Kutta:
x_new = x_old + h/6 * (k1 + 2k2 + 2k3 + k4)

k1 = f(t_old,x_old)
k2 = f(t_old+h/2, x_old+h*(k1/2))
k3 = f(t_old+h/2,x_old+h*(k2/2))
k4 = f(t_old+h,x_old +h*k3)

To solve second order ODEs, split the ODE into first order form then use vectors to hold the (x_old,y_old,t_old)variables
so: 
initial conditions x,y = [x0,y0]

'''

def euler_step(f,x0,t0,h):
    x1 = x0 + h*f(t0,x0)
    t1 = t0+h
    return x1,t1

def rk4_step(f,x0,t0,h):
    k1 = f(t0,x0)
    k2 = f(t0+h/2, x0+h*(k1/2))
    k3 = f(t0+h/2,x0+h*(k2/2))
    k4 = f(t0+h,x0+h*k3)
    x1 = x0 + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    t1 = t0+h
    return x1,t1

def system_of_odes(t,x):
    dxdt = x[1]
    dydt = -x[0]
    return np.array([dxdt, dydt])

def dx_dt(t,x):
    dx_dt = x
    return dx_dt

def true_solution(t):
    return np.exp(t)


def solve_to(f,x0,t0,tf,h, method):
    #:Params: f = function of system of ODEs
    #:Params: x0 = array of initial conditions
    #:Params: t0 = initial time 
    #:Params: tf = finishing time
    #:Params: h = step size
    #:Params: method = euler or rk4
    N = int((tf-t0)/h)
    dim = len(x0)
    x = np.zeros((N+1,dim))
    t = np.linspace(t0,tf,N+1)
    x[0] = x0
    for i in range(N):
        if method == 'euler':
            x[i+1],t[i+1] = euler_step(f, x[i], t[i], h)
        elif method == 'rk4':
            x[i+1],t[i+1] = rk4_step(f, x[i], t[i], h)
        else:
            raise ValueError("Invalid method. Use 'euler' or 'rk4'")
    return x,t

'''
#Finding Errors
timestep_values = np.logspace(-6, -1, 20) #e-6 to e-1
errors_euler = []
errors_rk4 = []
for h in timestep_values:
    x,t = solve_to(dx_dt,[1],0,3,h,method='euler')
    true_values = true_solution(t)
    error = np.abs(true_values - x)
    max_error = max(error)
    errors_euler.append(max_error)

for h in timestep_values:
    x,t = solve_to(dx_dt,[1],0,3,h,method='rk4')
    true_values = true_solution(t)
    error = np.abs(true_values - x)
    max_error = max(error)
    errors_rk4.append(max_error)

plt.loglog(timestep_values, errors_euler, label='euler', marker='o',color='red')
plt.loglog(timestep_values,errors_rk4,label='rk4', marker='o')
plt.xlabel('Max Errors')
plt.ylabel('Time Steps')
plt.legend()
plt.show()
'''


#Plots x and y against t

h = 0.5
tf = 50
t0 = 0
x0 = [1,0]
x, t = solve_to(system_of_odes,x0,t0,tf,h,method='euler')

plt.plot(t,x[:,0],label='x')
plt.plot(t,x[:,1],label='y')

dxdt = np.gradient(x[:, 0],t)
dydt = np.gradient(x[:,1],t)

plt.plot(x[:,0],dxdt)
plt.plot(x[:,1],dydt)
plt.xlabel('Xdot')
plt.ylabel('x')
#plt.savefig('Question3.jpeg')
plt.show() 
#x and xdot converges to 0
#As you increase h and tf, a runtime overflow error occurs in the rk4 method
#Using the euler method we can see a conversion to 0







