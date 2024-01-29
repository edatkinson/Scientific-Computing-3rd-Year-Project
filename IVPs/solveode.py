#!/usr/bin/env python
#
#  Script to solve ODE and plot the results.
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def main(filename=None):
    """Solve x'' = -x and plot to screen or file

    Run this as:

    $ ./solveode.py
    $ ./solveode.py myfig.pdf
    """
    # Generate the solution
    t, x, v = solve_ode()

    # Create a figure showing the solution
    fig = plot_solution(t, x, v)

    if filename is None:
        # $ ./solveode.py
        # (show on screen)
        plt.show()
    else:
        # $ ./solveode.py myfig.pdf
        # (save to file)
        fig.savefig(filename)


def solve_ode():
    """Generate numerical solution for x'' = -x

    Returns (t, x, v) ready for plotting.
    """
    #
    # The ODE and ICs give the initial value problem (IVP)
    #
    #   x'' = -x,  x(0) = 0, x'(0) = 1
    #
    # which has the unique solution
    #
    #   x = sin(t)
    #
    # To solve numerically we convert ODE to first order form as
    #
    #   x' = v
    #   v' = -x
    #
    # with the initial conditions
    #
    #   x(0) = 0
    #   v(0) = 1
    #
    # Now treating this as a vector ODE we have X = (x, v) and
    #
    #    (x, v)' = (v, -x)
    #
    # We need to make a function f that can compute the rhs from the vector
    # (x, v):
    #
    #   X' = f(X, t)

    def f_shm(X, t):
        x, v = X
        dxdt = v
        dvdt = -x
        dXdt = [dxdt, dvdt]
        return dXdt

    #
    # Initial conditions as a vector
    #
    x0 = 0
    v0 = 1
    X0 = [x0, v0]

    # Solve from t=0 to t=10 and get 200 equally spaced points in the output
    t = np.linspace(0, 10, 200)

    # Actually compute the solution:
    X_solution = odeint(f_shm, X0, t)

    # X_solution is a matrix with 200 rows and 2 columns. The first column is
    # x and the other is v.
    x_solution = X_solution[:, 0]
    v_solution = X_solution[:, 1]

    return t, x_solution, v_solution


def plot_solution(t, x, v):
    """Produce a figure with timeseries and phasespace plots"""

    # Create a figure with two plotting axes side by side:
    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_axes([0.58, 0.15, 0.35, 0.7])
    ax2 = fig.add_axes([0.08, 0.15, 0.35, 0.7])

    # Timeseries plot
    ax1.set_title('Time series: $x, v$ against $t$')
    ax1.plot(t, x, color='green', linewidth=2, label=r'$x$')
    ax1.plot(t, v, color='blue', linewidth=2, label=r'$v$')
    ax1.set_yticks([-1, 0, 1])
    ax1.set_xlabel(r'$t$')
    ax1.set_xticks([0, np.pi, 2*np.pi, 3*np.pi])
    ax1.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$'])
    ax1.grid()
    ax1.legend()

    # Phasespace plot
    ax2.set_title('Phase space: $v$ against $x$')
    ax2.plot(x, v, linewidth=2, color='red')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$v$', rotation=0)
    ax2.set_xticks([-1, 0, 1])
    ax2.set_yticks([-1, 0, 1])
    ax2.grid()

    # Return the figure handle for showing/saving
    return fig


if __name__ == "__main__":
    #
    # If run as a command line script (rather than imported) then we call main
    # passing the command line arguments.
    #
    import sys
    args = sys.argv[1:]
    main(*args)
