## Exercise 1:Create a file `ode_solver.py`
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fsolve
from typing import Callable, Tuple
import os


# 1. Add the function `Euler` for the Euler methods. 
def Euler(f: Callable, t0: float, t_end: float, y0: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves an initial value problem y' = f(t, y) using the Euler method.

    Args:
        f (function): The function that computes the derivative, f(t, y).
        t0 (float): Initial x-value.
        t_end (float): The final x-value to solve up to.
        y0 (np.array): Initial state vector y(t0).
        h (float): The step size.

     Returns:
        (np.array, np.array):t_values, y_values (2D array)
    """
    n = int((t_end - t0)/h) + 1
    tvalues = np.linspace(t0, t_end, n)

    y0 = np.asarray(y0)
    yvalues = np.zeros((n, len(y0)))
    yvalues[0] = y0
    
    for i in range(n-1):
        y_prime = f(tvalues[i], yvalues[i])
        yvalues[i+1] = yvalues[i] + h * y_prime
    return tvalues,  yvalues
    
# 2. Add  the function `rungekutta4` for the Runge-Kutta method of order 4
def rungekutta4(f: Callable, t0: float, t_end: float, y0: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves an initial value problem y' = f(t, y) using the Runge-Kutta 4th method.

    Args:
        f (function): The function that computes the derivative, f(t, y).
        t0 (float): Initial x-value.
        t_end (float): The final x-value to solve up to.
        y0 (np.array): Initial state vector y(t0).
        h (float): The step size.

    Returns:
        (np.array, np.array): t_values, y_values (2D array)
    """
    n = int((t_end - t0)/h) + 1
    t_values = np.linspace(t0, t_end, n)
    
    y0 = np.asarray(y0)
    y_values = np.zeros((n, len(y0)))
    y_values[0] = y0
    
    for i in range(n-1):
        t = t_values[i]
        y = y_values[i]
        
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        
        y_values[i+1] = y + (k1 + 2*k2 + 2*k3 + k4)/6.0
    return t_values,  y_values

def Euler_Backwards(f: Callable, t0: float, t_end: float, y0: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a system of ODEs using the Backward/Implicit Euler Method

    Args:
        f (function): The function that computes the derivative, f(t, y).
        t0 (float): Initial t-value.
        t_end (float): The final x-value to solve up to.
        y0 (np.array): Initial state vector, corresponding to y(t0).
        h (float): The step size.

    Returns:
        (np.array, np.array): tvalues, yvalues (2D array)
    """
    n = int((t_end - t0)/h) + 1
    tvalues = np.linspace(t0, t_end, n)

    y0 = np.asarray(y0)
    yvalues = np.zeros((n, len(y0)))
    yvalues[0] = y0
    
    for i in range(n-1):
        tn = tvalues[i]
        yn = yvalues[i]
        tn1 = tvalues[i+1]

        # 0 = yn1 - yn - h * f(tn1, yn1)
        def solving(yn1):
            return yn1 - h * f(tn1, yn1) - yn

        yn1_guess = yn + h * f(tn, yn)
        yn1 = fsolve(solving, yn1_guess) # fsolve finds the root of yn1
        yvalues[i+1] = yn1
    return tvalues,  yvalues
    
# 3. Add the function `TestEuler` that tests your Euler function
def TestEuler() -> None:
    print("-- Testing Euler's Method Convergence Order --")

    f = lambda t, y: -y[0] * np.sin(t)
    y_exact_func = lambda t: np.exp(np.cos(t) - 1)

    t0, t_end, y0 = 0, 8, np.array([1.0])
    hvalues = np.array([0.1, 0.05, 0.025, 0.01, 0.005])
    errors = []

    print(f"Test Problem: y' = -y*sin(t) on [{t0}, {t_end}], y(0)=1, Exact: {y_exact_func(t_end):.6f}")
    for h in hvalues:
        tvals, yvals, = Euler(f, t0, t_end, y0, h)
        y_approx = yvals[-1, 0]
        y_exact = y_exact_func(t_end)
        error = np.abs(y_approx-y_exact)
        errors.append(error)
        
        print(f"  h = {h:<7} | Approx at t={t_end}: {y_approx:.6f}| Abs Error: {error:.6e}")

    # Calculating order form log-log plot
    log_h = np.log(hvalues)
    log_err = np.log(errors)
    slope, _ = np.polyfit(log_h, log_err, 1)

    print(f"\nCalculated Convergence Order (slope): {slope:.4f}")
    assert 0.9 < slope < 1.1, f"Euler order ({slope:.4f}) not close to 1."
    print("Test Passed: Order is approximately 1.")


    plot_dir = 'plots_Euler_RK4_ImplicitEuler'
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(log_h, log_err, 'bo-', label=f'Slope = {slope:.4f}')
    plt.plot(log_h, 1.0 * log_h + log_err[0] - log_h[0], 'r--', label='Ideal Order 1')
    
    plt.xlabel('log(h)')
    plt.ylabel('log(Error)')
    plt.title("Euler Method Convergence (f = -y*sin(t))")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(plot_dir, 'euler_convergence.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved convergence plot to '{save_path}'\n")
    
# 4. Add the function `TestRungeKutta4` that tests your Runge-Kutta function of order 4
def TestRungeKutta4() -> None:
    print("-- Testing Runge-Kutta 4th Order Method Convergence Order --")
    
    f = lambda t, y: -y[0] * np.sin(t)
    y_exact_func = lambda t: np.exp(np.cos(t) - 1)

    t0, t_end, y0= 0, 8, np.array([1.0])
    
    hvalues = np.array([0.1, 0.05, 0.025, 0.01, 0.005])
    errors = []

    print(f"Test Problem: y' = -y*sin(t) on [{t0}, {t_end}], y(0)=1, Exact: {y_exact_func(t_end):.6f}")
    for h in hvalues:
        tvals, yvals, = rungekutta4(f, t0, t_end, y0, h)
        y_approx = yvals[-1, 0]
        y_exact = y_exact_func(t_end)
        error = np.abs(y_approx-y_exact)
        errors.append(error)

        print(f"  h = {h:<7} | Approx at t={t_end}: {y_approx:.6f}| Abs Error: {error:.6e}")

    # Calculating order form log-log plot
    log_h = np.log(hvalues)
    log_err = np.log(errors)
    slope, _ = np.polyfit(log_h, log_err, 1)

    print(f"\nCalculated Convergence Order (slope): {slope:.4f}")
    assert 3.9 < slope < 4.1, f"RK4 order ({slope:.4f}) not close to 4."
    print(f"Test Passed: Order is approximately 4.")

    plot_dir = 'plots_Euler_RK4_ImplicitEuler'
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(log_h, log_err, 'bo-', label=f'Slope = {slope:.4f}')
    # *** FIXED: Ideal order line was 1.0, should be 4.0 ***
    plt.plot(log_h, 4.0 * log_h + log_err[0] - 4.0 * log_h[0], 'r--', label='Ideal Order 4')
    
    
    plt.xlabel('log(h)')
    plt.ylabel('log(Error)')
    plt.title("RK4 Method Convergence (f = -y*sin(t))")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(plot_dir, 'rk4_convergence.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved convergence plot to '{save_path}'\n")
    
def TestEulerBackwards() -> None:
    print("-- Testing Implicit Euler's Method Convergence Order --")

    f = lambda t, y: -y[0] * np.sin(t)
    y_exact_func = lambda t: np.exp(np.cos(t) - 1)

    t0, t_end, y0 = 0, 8, np.array([1.0])
    hvalues = np.array([0.1, 0.05, 0.025, 0.01, 0.005])
    errors = []

    print(f"Test Problem: y' = -y*sin(t) on [{t0}, {t_end}], y(0)=1, Exact: {y_exact_func(t_end):.6f}")
    for h in hvalues:
        tvals, yvals, = Euler_Backwards(f, t0, t_end, y0, h)
        y_approx = yvals[-1, 0]
        y_exact = y_exact_func(t_end)
        error = np.abs(y_approx-y_exact)
        errors.append(error)

        print(f"  h = {h:<7} | Approx at t={t_end}: {y_approx:.6f}| Abs Error: {error:.6e}")

    # Calculating order form log-log plot
    log_h = np.log(hvalues)
    log_err = np.log(errors)
    slope, _ = np.polyfit(log_h, log_err, 1)

    print(f"\nCalculated Convergence Order (slope): {slope:.4f}")
    assert 0.9 < slope < 1.1, f"Implicit Euler order ({slope:.4f}) not close to 1."
    print("Test Passed: Order is approximately 1.")


    plot_dir = 'plots_Euler_RK4_ImplicitEuler'
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(log_h, log_err, 'bo-', label=f'Slope = {slope:.4f}')
    plt.plot(log_h, 1.0 * log_h + log_err[0] - log_h[0], 'r--', label='Ideal Order 1')
    
    plt.xlabel('log(h)')
    plt.ylabel('log(Error)')
    plt.title("Implicit Euler Method Convergence (f = -y*sin(t))")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(plot_dir, 'implicit_euler_convergence.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved convergence plot to '{save_path}'\n")
    
'''
5. Add both tests to
   if __name__ == '__main__':
    # run the test
'''

if __name__ == '__main__':
    print("--- Runninng Solver Tests ---")

    try:
        TestEuler()
        TestRungeKutta4()
        TestEulerBackwards()
        print("\nAll tests passed successfully!")
        
    except AssertionError as e:
        print(f"\n--- TEST FAILED ---")
        print(e)
    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(e)