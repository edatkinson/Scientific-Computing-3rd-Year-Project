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

def euler_step(f,x,t,h):
    x = x+ h*f(t,x)
    return x

def dx_dt(t,x):
    dx_dt = x
    return dx_dt


t0 = 0
x0 = 1
tf = 2
h = 0.1

#time stepping
t = t0
x = x0

while t < tf:
    x = euler_step(dx_dt, x, t, h)
    t+=h
    



