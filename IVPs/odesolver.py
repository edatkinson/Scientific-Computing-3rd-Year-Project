import numpy as np 
import matplotlib.pyplot as plt 




def main(filename=None):
    t = np.linspace(0,50,100)
    x0 = np.array([1,0])
    x, y, t = solve_ode(system_of_odes,x0,t,method='rk4',h=0.1)
    fig = plot_solution(t,x,y)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)

def euler_step(f,x0,t0,h):
    x1 = x0 + h*f(t0,x0)
    t1 = t0+h
    return x1

def rk4_step(f,x0,t0,h):
    k1 = f(t0,x0)
    k2 = f(t0+h/2, x0+h*(k1/2))
    k3 = f(t0+h/2,x0+h*(k2/2))
    k4 = f(t0+h,x0+h*k3)
    x1 = x0 + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x1


def system_of_odes(t,x):
    dxdt = x[1]
    dydt = -x[0]
    return np.array([dxdt, dydt])

def dx_dt(t,x):
    dx_dt = x
    return dx_dt

def true_solution(t):
    return np.exp(t)


# def solve_to(f,x0,t0,tf,h, method):
#     #:Params: f = function of system of ODEs
#     #:Params: x0 = array of initial conditions
#     #:Params: t0 = initial time 
#     #:Params: tf = finishing time
#     #:Params: h = step size or delta_max
#     #:Params: method = euler or rk4
#     N = int((tf-t0)/h)
#     dim = len(x0)
#     x = np.zeros((N+1,dim))
#     t = np.linspace(t0,tf,N+1)
#     x[0] = x0
#     for i in range(N):
#         if method == 'euler':
#             x[i+1] = euler_step(f, x[i], t[i], h)
#         elif method == 'rk4':
#             x[i+1] = rk4_step(f, x[i], t[i], h)
#         else:
#             raise ValueError("Invalid method. Use 'euler' or 'rk4'")
#     return x

def solve_to(func, x0, t0, tf, h, method):
    t = np.arange(t0, tf, h) #Create an array of times from t0 to tf
    sol = np.zeros((len(t), len(x0))) #Create an array to hold the solution
    sol[0] = x0 #Set the initial conditions

    for i in range(len(t)-1):
        if method == 'euler':
            sol[i+1] = euler_step(func, sol[i], t[i], h)
        elif method == 'rk4':
            sol[i+1] = rk4_step(func, sol[i], t[i], h)
        else:
            raise ValueError("Invalid method. Use 'euler' or 'rk4'")

    return sol[-1] #Return only the final state



'''
#Finding Errors
timestep_values = np.logspace(-6, -1, 20) #e-6 to e-1
errors_euler = []
errors_rk4 = []
for h in timestep_values:
    x,t = solve_to(dx_dt,[1],0,3,h,method='euler')
    true_values = true_solution(t)
    error = np.abs(true_values[-1] - x[-1,0]) 
    errors_euler.append(error)

for h in timestep_values:
    x,t = solve_to(dx_dt,[1],0,3,h,method='rk4')
    true_values = true_solution(t)
    error = np.abs(true_values[-1] - x[-1,0])
    errors_rk4.append(error)

plt.loglog(timestep_values, errors_euler, label='euler', marker='o',color='red')
plt.loglog(timestep_values,errors_rk4,label='rk4', marker='o')
plt.ylabel('Max Errors')
plt.xlabel('Time Steps')
plt.legend()
plt.show()
'''




def solve_ode(func, x0, t, method='rk4', h=1):
   
    sol = np.zeros((len(t), len(x0))) #Create an array to hold the solution
    sol[0] = x0 #Set the initial conditions
    
    for i in range(len(t)-1):
        sol[i+1] = solve_to(func, sol[i], t[i], t[i]+h, h, method) #Solve the ODE at each time step
    
    x_sol = sol[:, 0] #Extract the x values
    y_sol = sol[:, 1] #Extract the y values
    t_sol = t #Extract the time values (array of times)
    return  x_sol, y_sol, t_sol


def plot_solution(t, x, v):
    """Produce a figure with timeseries and phasespace plots"""

    # Create a figure with two plotting axes side by side:
    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_axes([0.58, 0.15, 0.35, 0.7])
    ax2 = fig.add_axes([0.08, 0.15, 0.35, 0.7])

    # Timeseries plot
    ax1.set_title('Time series: $x, y$ against $t$')
    ax1.plot(t, x, color='green', linewidth=2, label=r'$x$')
    ax1.plot(t, v, color='blue', linewidth=2, label=r'$y$')
    ax1.set_yticks([-1, 0, 1])
    ax1.set_xlabel(r'$t$')
    ax1.set_xticks([0, np.pi, 2*np.pi, 3*np.pi])
    ax1.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$'])
    ax1.grid()
    ax1.legend()

    # Phasespace plot
    ax2.set_title('Phase space: $y$ against $x$')
    ax2.plot(x, v, linewidth=2, color='red')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$y$', rotation=0)
    ax2.set_xticks([-1, 0, 1])
    ax2.set_yticks([-1, 0, 1])
    ax2.grid()

    # Return the figure handle for showing/saving
    return fig


'''
if __name__ == "__main__":
    #
    # If run as a command line script (rather than imported) then we call main
    # passing the command line arguments.
    #
    import sys
    args = sys.argv[1:]
    main(*args)
'''



#Plots x and y against t

h = 0.1
t = np.linspace(0,50,100)
x0 = np.array([1,0])
x, y, t = solve_ode(system_of_odes,x0,t,method='rk4',h=h)

plt.plot(t,x,label='x')
plt.plot(t,y,label='y')

# dxdt = np.gradient(x[:, 0],t)
# dydt = np.gradient(x[:,1],t)

# plt.plot(x[:,0],dxdt)
# plt.plot(x[:,1],dydt)
# plt.xlabel('Xdot')
# plt.ylabel('x')
#plt.savefig('Question3.jpeg')
plt.show() 
#x and xdot converges to 0
#As you increase h and tf, a runtime overflow error occurs in the rk4 method
#Using the euler method we can see it converging to 0







