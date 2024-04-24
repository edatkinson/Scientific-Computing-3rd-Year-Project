# Scientific Computing

This repository contains a suite of numerical methods designed to solve ordinary differential equations (ODEs) and boundary value problems (BVPs).

## Key Components

### ODE Solver (`solve_ode`)

- Implements Euler's method and the 4th order Runge-Kutta (RK4) method.

### Numerical Shooting (`bvp_and_shooting`)

- Solves periodic solutions to ODEs by converting BVPs into initial value problems.
- Employs `fsolve` from `scipy.optimize` for effective root finding.
- Includes functions like `orbit` and `limit_cycle_finder` to trace and locate limit cycles.

### Numerical Continuation

- Offers methods for both Natural and Pseudo Arc-Length (PAL) continuation.

### Sparse and Dense BVP Solver (`sparse_dense_bvp`)

- Supports both sparse and dense matrix operations, selectable based on problem specifics and computational resources.

### Time variant Diffusive PDE solver (`Diffusion_OO`)

- Solves time variant PDEs for Dirichlet, Neumann and Robin BCs using a range of methods and storage types

## Usage

To use these methods, ensure you have Python installed along with necessary packages like `numpy`, `scipy`, and `logging`. Refer to the individual function docstrings for specific examples and parameter details.

## Testing
- Most of the testing is located in `tests.py`, other tests include `shooting_test.py` and `limit_cycle_comparison.py`
  
## Contribution

Contributions to improve the methods or extend the range of solvable equations are welcome. Please ensure to follow the existing coding and documentation standards.

