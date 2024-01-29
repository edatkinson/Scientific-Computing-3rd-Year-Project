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

def dx_dt(t,x):
    dx_dt = x
    return dx_dt

def true_solution(t):
    return np.exp(t)


def solve_to(f,x0,t0,tf,h, method):
    N = int((tf-t0)/h)
    #error = np.zeros(N+1)
    x = np.zeros(N+1)
    t = np.linspace(t0,tf,N+1)
    #solution = x0*np.exp(t) 
    x[0] = x0
    #solution[0] = x0
    for i in range(N):
        if method == 'euler':
            x[i+1],t[i+1] = euler_step(f, x[i], t[i], h)
            #error[i] = abs(-solution[i]+x[i])
        elif method == 'rk4':
            x[i+1],t[i+1] = rk4_step(f, x[i], t[i], h)
            #error[i] = abs(-solution[i] + x[i])
        else:
            raise ValueError("Invalid method. Use 'euler' or 'rk4'")
    

    return x,t #,error


timestep_values = np.logspace(-3, -1, 20)
errors = []
for h in timestep_values:
    x,t = solve_to(dx_dt,1,0,3,h,method='rk4')
    true_values = true_solution(t)
    error = np.abs(true_values - x)
    max_error = max(error)
    errors.append(max_error)

plt.loglog(timestep_values, errors, label='Error', marker='o')
plt.show()







