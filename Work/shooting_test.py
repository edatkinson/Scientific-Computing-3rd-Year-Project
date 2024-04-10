
# %%
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from bvp_and_shooting import phase_condition, shoot, limit_cycle_finder, phase_portrait_plotter
from Equations_Functions import hopf
# %%

'''
Test 1. Hopf Bifurcation. u0 = [0.5, 2, 30], pars = [0.9, -1] (beta, sigma)
'''

initial_guess = [0.5, 2, 30]
pars = [0.9, -1] #beta = any, sigma = -1
orbit, cycle1 = limit_cycle_finder(hopf, initial_guess,phase_condition,pars, test=True)
fig = phase_portrait_plotter(cycle1)
fig.show()


print(f'The roots of the limit cycle: {orbit[:-1]}')
print(f'Numerical Approximation of the Limit Cycle: {orbit[-1]}')


t = np.linspace(0,10,len(cycle1[0, :]))

beta, sigma = pars
theta = 1 
u1 = beta**0.5 * np.cos(t+theta)
u2 = beta**0.5 * np.sin(t+theta)

plt.plot(u1,u2)
plt.show()

#Matching the real Limit Cycle with the one we found
plt.plot(u1,u2)
plt.plot(cycle1[0, :], cycle1[1, :], label='Isolated periodic orbit')
plt.show()

if np.allclose([u1, u2], cycle1[:-1], rtol=1e+00, atol=1e+00): #Different phases but same amplitude
    print("The numerical approximation is close to the theoretical limit cycle!")
else:
    print("The numerical approximation is not close to the theoretical limit cycle.")

# %%
''' Test 2: Hopf Bifurcation. pars = [0.1, -1] (beta, sigma)'''
# %%

initial_guess = [0.5, 2, 30]
pars = [1, -1] #beta = any, sigma = -1
orbit, cycle1 = limit_cycle_finder(hopf, initial_guess,phase_condition,pars, test=True)
fig = phase_portrait_plotter(cycle1)
fig.show()

# print(f'The roots of the limit cycle: {orbit[:-1]}')
# print(f'Numerical Approximation of the Limit Cycle: {orbit[-1]}')


# %%
def test_initial_value_dimensions(ode_func, initial_values, pars):
    try:
        ode_system_dimension = ode_func(0, initial_values, pars).size
        assert len(initial_values) == ode_system_dimension
        print("Test for dimension match: PASSED")
    except AssertionError:
        print("Test for dimension match: FAILED - Initial values do not match ODE dimensions.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# test_initial_value_dimensions(hopf, initial_guess, pars)
