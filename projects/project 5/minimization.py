## Exercise 1

## Create a file `minimization.py`

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Callable, Tuple, List
import os


# 1. Add the function `GradientDescent` for the gradient descent methods. 
def GradientDescent(DF: Callable, x0: np.ndarray, a: float, max_iter: int, error: float) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """
    Finds the local minimum of a function using the Gradient Descent method.

    Args:
        DF (Callable): The derivative of the function to minimize, f'(x).
        x0 (np.ndarray): The initial guess for the minimum.
        a (float): The learning rate (step size).
        max_iter (int): The maximum number of iterations to perform.
        tol (float): The tolerance for convergence (step size threshold).

    Returns:
        (np.ndarray, int, np.ndarray): The estimated minimum x-value, the number of iterations performed, and the history of the values obtained after each iteration.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()] # I will be using this to calculate the error so I can later on plot the errors vs iterations
    
    for i in range(max_iter):
        grad = DF(x)
        xnew = x - a*grad # calculating the new positions on the gradient

        history.append(xnew.copy())

        # checking for Convergence
        # # np.linalg.norm calculates the distance between the old and new point (substitute np.abs)
        if np.linalg.norm(xnew - x) < error:
            return xnew, i+1, history
            
        if np.linalg.norm(grad) < error:
            return xnew, i+1, history
        
        x = xnew
        
    return xnew, max_iter, history
    
# 2. Add the function `TestGradientDescent` that test your function
def TestGradientDescent() -> None:
    print("-- Testing Gradient Descent Convergence Order --\n")
    
    # Function: f(x,y) = x^2 + y^2
    # f = lambda v: np.array([v[0]**2, v[1]**2])
    # Gradient = [2x, 2y]
    df_2d = lambda v: np.array([2*v[0], 2*v[1]])

    # initial guess
    x0_y0 = np.array([-4.0, 3.5]) 
    min_2d, iters, history = GradientDescent(df_2d, x0_y0, a=0.1, max_iter=100, error=1e-6)

    print(f"Test Problem: f(x,y) = x^2 + y^2, Start: {x0_y0}")
    print(f"-> Converged to: {min_2d}")
    print(f"-> {iters} iterations\n")
    print(f"Expected: [0. 0.]")

    expected = np.array([0.,0.])
    abs_error = np.linalg.norm(min_2d - expected)
    print(f"Absolute Error: {abs_error:.6e}")

    assert abs_error < 1e-4, f"Gradient Descent failed to converge to (-4.0, 3.5). Got {min_2d}"
    print("Test Passed: Minimum found correctly.")

    #plotting errors
    plot_dir = 'minimization_plot'
    os.makedirs(plot_dir, exist_ok=True)
    
    errors = [np.linalg.norm(x - expected) for x in history]
    plt.figure(figsize=(8, 6))

    plt.plot(range(len(errors)), errors, marker='o', linestyle='-', color='r', label="Error Magnitude")
    
    plt.title("Error vs. Iteration")
    plt.xlabel("Iteration Number")
    plt.ylabel("Absolute Error (Distance to 0,0)")
    plt.grid(True)
    plt.legend()
    
    save_path = os.path.join(plot_dir, 'minimization_error.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved convergence plot to '{save_path}'\n")

'''
3. Add the test to
   if __name__ == '__main__':
    # run the test
'''
if __name__ == '__main__':
    print("--- Runninng Solver Test ---")

    try:
        TestGradientDescent()
        print("\nTest passed successfully!")
        
    except AssertionError as e:
        print(f"\n--- TEST FAILED ---")
        print(e)
    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(e)