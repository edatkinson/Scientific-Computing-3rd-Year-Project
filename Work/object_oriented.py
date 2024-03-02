
from bvp_and_shooting import lokta_volterra, hopf
from scipy.optimize import fsolve
import numpy as np

class ODESystem:
    def __init__(self, function, params):
        self.function = function
        self.params = params

    def ode(self, t, U):
        return self.function(t, U, self.params)
#ODESysetm creates objects of the class ODESystem which has the function and parameters as attributes
lokta_volterra_system = ODESystem(lokta_volterra, [1,1,1])


class NumericalMethod:
    def __init__(self, ode_system, initial_conditions, duration):
        self.ode_system = ode_system
        self.initial_conditions = initial_conditions
        self.duration = duration
    
    def integrate(self):
        t = np.linspace(0,self.duration,150)
        sol,_ = solve_ode(self.ode_system,self.initial_conditions[:-1],t,'rk4',0.01)
        return sol[-1,:]

    def phase_condition(self):
        return np.array([self.ode_system.ode(0,self.initial_conditions)[0]])
    
    def shoot(self):
        u0 = self.initial_conditions[:-1]
        T = self.initial_conditions[-1]
        array = np.hstack((u0 - self.integrate(), self.phase_condition()))
        return array
    def roots(self, estimate):
        result = fsolve(lambda estimate: self.shoot(self.ode_system,estimate,self.phase_condition),estimate)
        return result 

duration = 20
lokta_volterra_shooting = NumericalMethod(lokta_volterra_system, lokta_volterra_system.params, duration)

print(lokta_volterra_shooting.roots([1,1,1]))

    
