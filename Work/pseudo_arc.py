

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# Define the modified Hopf bifurcation ODE system
def modified_hopf(t, u, pars):
    beta = pars[0]
    u1, u2 = u
    du1dt = beta * u1 - u2 + u1 * (u1**2 + u2**2) - (u1 * (u1**2 + u2**2)**2)
    du2dt = u1 + beta * u2 + u2 * (u1**2 + u2**2) - (u2 * (u1**2 + u2**2)**2)
    return np.array([du1dt, du2dt])

# Define the phase condition for the limit cycle
def phase_condition(ode, u0, pars):
    # The phase condition is the time derivative at t=0 being equal to zero for the first state variable
    return np.array([ode(0, u0, pars)[0]])

# Define the shooting function for the limit cycle
def shoot(ode, phase_cond, u0, T, pars):
    def boundary_condition(u0):
        sol = solve_ivp(lambda t, y: ode(t, y, pars), (0, T), u0, method='RK45', rtol=1e-6)
        final_state = sol.y[:, -1][-1]  # Extracting the last value of sol.y
        phase_condition_val = phase_cond(ode, u0, pars)
        return np.append(final_state - u0, phase_condition_val)

    return boundary_condition

# Define the function to find the limit cycle using the shooting method
def limit_cycle_finder(ode, phase_cond, u0_initial, T_initial, pars):
    u0_T_initial = np.append(u0_initial, T_initial)
    sol = fsolve(shoot(ode, phase_cond, u0_initial, T_initial, pars), u0_T_initial, xtol=1e-6, epsfcn=1e-6)
    return sol[:-1], sol[-1]  # Return the found limit cycle initial conditions and period

# Define the function for natural parameter continuation
def natural_parameter_continuation(ode, phase_cond, u0_initial, T_initial, pars_initial, beta_range, steps):
    beta_values = np.linspace(beta_range[0], beta_range[1], steps)
    u0 = u0_initial
    T = T_initial
    pars = pars_initial
    limit_cycles = []

    for beta in beta_values:
        pars[0] = beta  # Update the parameter
        u0, T = limit_cycle_finder(ode, phase_cond, u0, T, pars)
        limit_cycles.append((u0, T))

    return beta_values, limit_cycles

# Define parameters for the continuation
u0_initial = np.array([1.0, 0.0])  # Initial guess for state variables
T_initial = 2 * np.pi  # Initial guess for the period of the limit cycle
pars_initial = [0.1]  # Initial value of the parameter beta
beta_range = [0, 2]  # Range of beta values to track the limit cycle
steps = 50  # Number of steps in the continuation

# Perform the continuation
beta_values, limit_cycles = natural_parameter_continuation(modified_hopf, phase_condition, u0_initial, T_initial, pars_initial, beta_range, steps)

# Plot the results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(beta_values, [lc[0][0] for lc in limit_cycles], label='u1 of Limit Cycle')
plt.plot(beta_values, [lc[0][1] for lc in limit_cycles], label='u2 of Limit Cycle')
plt.title('Limit Cycle Continuation')
plt.xlabel('Beta')
plt.ylabel('State Variables of Limit Cycle')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(beta_values, [lc[1] for lc in limit_cycles], label='Period of Limit Cycle')
plt.title('Limit Cycle Period Continuation')
plt.xlabel('Beta')
plt.ylabel('Period')
plt.legend()

plt.tight_layout()
plt.show()
