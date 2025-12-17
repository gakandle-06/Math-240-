# Lab 4: Solving ODEs


## Exercise 1

Create a file `ode_solver.py`

1. Add the function `Euler` for the euler methods. 
2. Add  the function `rungekutta4` for the runge kutta method of order 4
3. Add the function `TestEuler` that test your Euler function
4. Add the function `TestRungeKutta4` that test your runge kutta function of order 4

5. Add both test to
```Python
   if __name__ == '__main__':
    # run the test
```

Now you can answer the following question in a jupyter notebook and by importing `ode_solver.py`

## Exercise 1

Use Euler/RK4's method to approximate/plot the solutions for each of the following initial-value problems. (Plot your solution for all the values of t)
1. $y'=3e^{-ty}$, $1\leq t \leq 3$, $y(0)=1/2$, with $h=0.005$
2. $y'=\frac{1+t^2}{1+2y}$, $2\leq t \leq 5$, $y(2)=3$, with $h=0.01$
3. $y' = t^{2}\sin(y)$, $0 \leq t \leq 4$, $y(0) = 2$, with $h=0.001$
4. $ y' = \frac{t^3}{y},\; 1 < t < 10, \; y(1) = 1$, with $h=0.002$

## Exercise 2

Use Euler/RK4's method to approximate/plot the solutions for each of the following initial-value problem.
$$ y'= \frac{2-2ty}{t^2+1}, ~~~~ 1\leq t \leq 4, ~~~~ y(1)=2$$
The actual solutions to the initial-value is 
$$y(t)=\frac{2t+2}{t^2+1}.$$
1. Compute/plot the error of your approximation and bound the error for $1\leq t \leq 4$, for $h=0.01, 0.005, 0.001$
2. Approximate $y(4)$ for for $h=0.01, 0.005, 0.001$ and compute the error.

## Exercise 3

Given the initial-value problem 
$$y'=te^{-t}-y,\; 0\leq t \leq 5, \; y(0)=1$$
Approximate y(5) using Euler/RK4's method with h = 0.1, h =0.01, and h = 0.001. 

### Exercise 4

Test the [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp)

## Exercise 5

Lokta-Volterra Equations:
$$\begin{cases}
 & \displaystyle{\frac{dx}{dt} = \alpha x - \beta xy } \\
 & \displaystyle{\frac{dy}{dt} = \delta xy - \gamma y}
 \end{cases}$$
 where
* x is the number of prey
* y is the number of some predator
* $\alpha$, $\beta$, $\gamma$, $\delta$ are positive real parameters describing the interaction of the two species.
* The prey are assumed to have an unlimited food supply and to reproduce exponentially, unless subject to predation; this exponential growth is represented in the equation above by the term $\alpha x$. 
* The rate of predation upon the prey is represented above by $\beta xy$. If either x or y is zero, then there can be no predation.
* $\delta xy$ represents the growth of the predator population.
* $\gamma y$ represents the loss rate of the predators due to either natural death or emigration, it leads to an exponential decay in the absence of prey


1. Solve the Lokta-Volterra equations with  the 4 different methods, with  $\alpha= 1/3$, $\beta = 3/4$, $\gamma = 1 = \delta$
2. Try another cool set of value and solve it.

## Exercise 6

Solve the pendulum equations
$$\begin{cases}
\frac{d\theta}{dt} &= \omega \\
\frac{d\omega}{dt} &=\frac{b}{m}\omega + \frac{g}{L}\sin(\theta)
\end{cases}$$
with 
* $L$ is the length of the pendulum
* $\theta$ is the angle
* $b$ is the damping coefficient
* $g$ is the gravity


![Alt text](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Faleksandarhaber.com%2Fwp-content%2Fuploads%2F2020%2F03%2Fpendulum1.jpg&f=1&nofb=1&ipt=66ff055ec65998da85ef499095f3b9d6602cec3fe7fb3c6b1550054043ef99ea&ipo=images)


For our case we can pick $L=1$, $b=0.05$, $g=9.81$, $m=1$. Start with $\theta=\pi/2$ and solve the ODE. Try with a different starting $\theta$ and solve the ODE.

## BONUS (Optional)

### Exercise 7

The SIR model. To learn more you can see [Compartemental Model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) or [SIR Modeling - Western Kentucky University
](https://people.wku.edu/lily.popova.zhuhadar/)

* $S(t)$: the number of individuals susceptible of contracting the infection at time $t$,
* $I(t)$: the number of individuals that are alive and infected at time t;
* $R(t)$: the cumulative number of individuals that recovered from the disease up to time t;

In addition, $N$ is the total number of people in the area at time $t$ with $N = S(t) + I(t) + R(t)$.
The SIR model is given by the following expressions:
$$
\begin{equation} 
\begin{split}
\dfrac{dS}{dt} &=  -\frac{\beta I S}{N}, \\ 
\dfrac{dI}{dt} &=  \frac{\beta I S}{N} - \gamma I\\
\dfrac{dR}{dt} &= \gamma I,\\
\end{split}
\end{equation}
$$
#### Part I
Pick $\beta=0.2$, $\gamma=1/10$, $N=1000$, $I(0)=1$, solve this epidemic problem. What is the percentage of Recovered/Immuned indivual needed for the desease to stop spreading?

#### Part II

Find and model the spread of specific desease/infection using the SIR model.


### Exercise 8

Use the Euler Implicit method to approximate/plot the solutions to each of the following initial-value. See [Euler Implicite](https://en.wikipedia.org/wiki/Backward_Euler_method).
1. $y' = -ty + 4t/y$, $0 \leq t \leq 1$, $y(0) = 1$, with $h = 0.1$
2. $y' = \frac{y^2+y}{t}$, $1 \leq  t  \leq 3$, $y(l) = -2$, with $h = 0.2$ 