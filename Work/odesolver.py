import numpy as np 
import matplotlib.pyplot as plt 
import sys

def main(filename=None, filename2=None):
    print(f"filename1: {filename}, filename2: {filename2}")
    t = np.linspace(0,20,100)
    x0 = np.array([1,0])
    x0_error = np.array([1])
    x, t = solve_ode(system_of_odes,x0,t,method='rk4',h=0.1)
    fig = plot_solution(t,x[:,0],x[:,1])
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)

    fig2 = plot_errors(dx_dt, true_solution, x0_error, t0=0, tf=2)
    if filename2 is None:
        plt.show()
    else:
        fig2.savefig(filename2)


def euler_step(f,x0,t0,h):
    x1 = x0 + h*f(t0,x0)
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

def solve_to(func, x0, t0, tf, deltat_max, method):
    """
    Numerically solves a differential equation using the specified method.
    
    Parameters:
    - func: The function to be solved, which takes time t and state x as inputs.
    - x0: The initial state.
    - t0: The initial time.
    - tf: The final time.
    - deltat_max: The maximum time step.
    - method: The numerical method to use ('euler' or 'rk4').
    
    Returns:
    - The final state of the system and the corresponding time.
    """
    # Adjusting the time steps to ensure the final time step lands exactly on tf
    num_steps = int(np.ceil((tf - t0) / deltat_max))
    actual_deltat = (tf - t0) / num_steps  # Adjust time step size
    t = np.linspace(t0, tf, num_steps + 1)  # Include tf in the array
    
    # Preallocate solution array
    sol = np.zeros((len(t),) + np.shape(x0))
    sol[0] = x0
    
    # Integration loop
    for i in range(num_steps):
        if method == 'euler':
            sol[i + 1] = euler_step(func, sol[i], t[i], actual_deltat)
        elif method == 'rk4':
            sol[i + 1] = rk4_step(func, sol[i], t[i], actual_deltat)
        else:
            raise ValueError("Invalid method. Use 'euler' or 'rk4'")
    
    return sol[-1], t[-1]

def solve_ode(func, x0, t, method, h):
    #Params:func = function to solve
    #Params: x0 = initial conditions
    #Params: t = array of times which you want to solve for
    #Params: method = 'euler' or 'rk4'
    #Params: h = time step
    #Returns: sol = solution array
    '''
    Could input tspan = [t0,tf] and use np.arange(t0,tf,h) to create the array of times
    '''
    sol = np.zeros((len(t), len(x0))) #Create an array to hold the solution
    sol[0] = x0 #Set the initial conditions
    
    for i in range(len(t)-1):
        sol[i+1], tf = solve_to(func, sol[i], t[i], t[i+1], h, method) #Solve the ODE at each time step
    
    # x_sol = sol[:, 0] #Extract the x values
    # y_sol = sol[:, 1] #Extract the y values
    t_sol = t #Extract the time values (array of times)
    return  sol, t_sol


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


def plot_errors(dx_dt, true_solution, x0, t0, tf):
    timestep_values = np.logspace(-6, -1, 10) #e-6 to e-1
    errors_euler = []
    errors_rk4 = []

    for h in timestep_values:
        x,t = solve_to(dx_dt,x0,t0,tf,h,'euler')
        true_values = true_solution(t)
        error = np.abs(true_values - x[-1])
        errors_euler.append(error)

    for h in timestep_values:
        x,t = solve_to(dx_dt,x0,t0,tf,h,'rk4')
        true_values = true_solution(t)
        error = np.abs(true_values - x[-1])
        errors_rk4.append(error)
    fig, ax = plt.subplots()
    ax.loglog(timestep_values,errors_euler, label='euler', marker='o',color='red')
    ax.loglog(timestep_values,errors_rk4,label='rk4', marker='o')
    ax.set_ylabel('Error')
    ax.set_xlabel('Time Steps')
    ax.legend()
    # fig.savefig('error_plot.jpeg')
    return fig




if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    main(*args)









