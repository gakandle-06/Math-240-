'''
Gavela Maculuve
Date: 27
## Exercise 1

Create a file roots.py where you will write two methods.

'''

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Callable, Tuple

'''
1. Write a function named `Newton` that takes as an argument a function, its
derivative, an initial guess, a tolerance, a maximum number of iterations, and return the zero of the function using Newton's method, the number of iterations it took, and the value of f at the root.
'''

# Newton Method
def Newton(f: Callable[[float], float], df: Callable[[float], float], x0: float, tol: float = 1e-8, max_iter: int = 150) -> Tuple[float,int, float]:
    '''
    Finds a root of a function using Newton's method
        
    Args:
        f (callable): The function;
        df (callable): The derivative of the function f;
        x0 (float): The initial guess;
        tol (float): tolerance for convergence (stops when |f(x_n)| < tol);
        max_iter (int): maximum number of iterations.
        
    Returns:
        Tuple:
        - xn (float): The zero of the function;
        - i (int): The number of iterations it took (int);
        - fxn (float): The value of f(x) at the found root (float).

    Raises:
        - ZeroDivisionError if derivatives are 0
        - ValueError if there is no convergence within max iterations
    '''
    xn = float(x0)
    for i in range(1, max_iter+1):
        fxn = f(xn)
        
        # check for convergence
        if abs(fxn) < tol:
            # print("Found Solution after", i, "iterations")
            return xn, i, fxn
            
        # checking for division by zero
        dfxn = df(xn)
        if dfxn == 0:
            raise ZeroDivisionError(f'Newton: Zero Division at iteration {i}')
        
        # Newton's Formula application
        xn = xn-fxn/dfxn
        
    raise ValueError(f'Did not converage after {max_iter} iterations.')

'''
2. Write a function named `Secant` that takes as an argument a function, its
derivative, an initial guess, a tolerance, a maximum number of iterations, and return the zero of the function using Secant's method
'''
def Secant(f: Callable[[float], float], x0: float, x1: float, tol: float = 1e-8, max_iter: int = 150) -> Tuple[float,int, float]:
    '''
    Finds a root of a function using the Secant method
        
    Args:
        f (callable): The function
        x0 (float): The initial guess
        x1 (float): The second guess
        tol (float): tolerance for convergence (stops when |x_next - x1| < tol)
        max_iter (int): maximum number of iterations
        
    Returns:
        Tuple:
            - xn (float): The zero of the function;
            - i (int): The number of iterations it took (int);
            - fxn (float): The value of f(x) at the found root (float).
            
    Raises:
        - ZeroDivisionError if derivatives are 0
        - ValueError if there is no convergence within max iterations
    '''
    f0 = f(x0)
    for i in range(1, max_iter+1):
        f1 = f(x1)
            
        if abs(f1-f0) < 1e-14:
            raise ZeroDivisionError(f'Zero Divison at iteration {i}')
        
        # Secant Formula application    
        xnext = x1 - f1 * (x1 - x0)/(f1-f0)
        
        #checking for convergence
        if abs(xnext - x1) < tol:
            # print("Found Solution After", i, "iterations")
            return xnext, i, f(xnext)
        
        # updating the points for the next iteration
        x0, x1 = x1, xnext
        f0 = f1
    
    raise ValueError(f'Did not converage after {max_iter} iterations.')

'''
3. Write a two-unit test for Newton (`TestNewton`) and Secant (`TestSecant`),
using a function you know the roots of, and more complex than a polynomial of degree 4

It should compute the convergence at three different points and make sure it decreases.
It should save the log-log plot of the convergence for each.
It should print the order of convergence. (slope of the log-log plot)
You could use `import unitest`
Start the unit test if the file is run directly: if __name__ == '__main__': run the test.
'''

def TestNewton() -> None:
    """
    Analyzes the convergence order of the Newton method and saves a plot.
    Uses f(x) = tan(x) - x.
    """
    print("----- Testing Newton -----")
    
    f = lambda x: np.tan(x) - x
    df = lambda x: np.tan(x)**2 # sec^2(x) - 1
    guess_points = [-4.4,-0.9, 0.2, 4.5]

    # plor directory
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    print("Calculating Convergence orders:")
    
    for x0 in guess_points:
        # true root
        true_root, _, _ = Newton(f, df, x0, tol=1e-15)

        
        errors = []
        xn = x0
        
        for _ in range(30):
            error = abs(xn - true_root)
            if error < 1e-15: break
            errors.append(error)
            
            fxn = f(xn)
            dfxn = df(xn)
            if dfxn == 0: break
            
            # Newton Method
            xn = xn - fxn/dfxn
        
        if len(errors) < 3:
            print(f"Skipping plot for x0={x0} (converged too fast)")
            continue
        
        log_errors = np.log(np.array(errors))
        log_errors_curr = log_errors[:-1]
        log_errors_next = log_errors[1:]
        
        slope, _ = np.polyfit(log_errors_curr, log_errors_next, 1)
        print(f"\tNewton convergence order from x0={x0}: {slope:.4f}")
        
        # Plot
        plt.plot(log_errors_curr, log_errors_next, 'o-', label=f'Start x0={x0}, order={slope:.2f}')
        assert np.all(np.diff(errors) < 0), "Newton errors did not decrease"

    plt.title('Newton Method Convergence Order (f=tan(x)-x)')
    plt.xlabel(r'log|error_{k}|')
    plt.ylabel(r'log|errorr_{k+1}|')
    plt.grid(True)
    plt.legend()
    save_path = os.path.join(plot_dir, 'newton_convergence.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Saved image of error slope to '{save_path}'")
    print("Newton Test Passed.")

def TestSecant() -> None:
    """
    Analyzes the convergence order of the Secant method and saves a plot.
    Uses f(x) = tan(x) - x.
    """
    print("\n--- Testing Secant ---")
    f = lambda x: np.tan(x) - x
    df = lambda x: np.tan(x)**2 # using to find trute root
    guess_points = [(-4.7, -4.4), (-1., -0.9), (0.2, 0.3), (4.5, 4.6)]

    # plot directory
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    print("Calculating Convergence orders: ")
    for x0_start, x1_start in guess_points:
        
        # true root (using Newton)
        true_root, _, _ = Newton(f, df, x0_start, tol=1e-15, max_iter=100)
        
        errors = []
        x0, x1 = x0_start, x1_start
        for _ in range(30): # Secant converges slower
            f0, f1 = f(x0), f(x1)
            
            if abs(f1 - f0) < 1e-14: break
            x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
            
            error = abs(x_next - true_root)
            if error < 1e-15: break
            errors.append(error)

            x0, x1 = x1, x_next
        
        if len(errors) < 3:
            print(f"Skipping plot for (x0,x1)=({x0:.4g},{x1:.4g}) (converged too fast)")
            continue

        log_errors = np.log(np.array(errors))
        log_errors_curr = log_errors[:-1]
        log_errors_next = log_errors[1:]
        
        slope, _ = np.polyfit(log_errors_curr, log_errors_next, 1)
        print(f"\tOrder of convergence from ({x0_start:.4e}, {x1_start:.4g}): {slope:.4f}")

        plt.plot(log_errors_curr, log_errors_next, 'o-', label=f'Start ({x0_start:.4e},{x1_start:.4e}), order={slope:.4f}')
        
        # Check if convergence decreases
        assert np.all(np.diff(errors) < 0), "Secant errors did not decrease"

    plt.title('Secant Method Convergence Order (f=tan(x)-x)')
    plt.xlabel(r'log|error_{k}|')
    plt.ylabel(r'log|errorr_{k+1}|')
    plt.grid(True)
    plt.legend()
    save_path = os.path.join(plot_dir, 'secant_convergence.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Saved image of error slope to '{save_path}'")
    print("Secant test passed.")


if __name__ == "__main__":
    print("--- Testing Root Finding Methods ---")
    
    try:
        TestNewton()
        TestSecant()
        
        print("\nAll tests passed successfully!")
        
    except AssertionError as e:
        print(f"\n--- TEST FAILED ---")
        print(e)
    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(e)