# Lab 5: Minimization


## Exercise 1

Create a file `minimization.py`

1. Add the function `GradientDescent` for the gradient descent methods. 
2. Add the function `TestGradientDescent` that test your Euler function
3. Add the test to
```Python
   if __name__ == '__main__':
    # run the test
```

Now you can answer the following question in a jupyter notebook and by importing `minization.py`

## Exercise 1

Compute the gradient of
$$ f(x,y,z) = x^2ye^{xz} +z\cos(xy)$$

## Exercise 2

Compute the first two step of the gradient descent by hand and find the best learning rate
$$f(x,y) = 2x^4y^2,\; x_0=1,y_0=2$$

## Exercise 3

Minimize the function below
$$D(x_1,x_2)=2x_1^2 + 2x_1x_2 + 2x_2^2 -\sqrt{6}x_1 $$


## Exercise 4

Minimize the function
$$ f(x_1,x_2) = (1-2x_2)^2+(x_1-2x_2^2)^2-2$$


## Exercise 5
Minimize the function
$$ f(x,y) = 2x^2–4xy+5y^2−4y+3$$

## Exercise 6

Assume you the following data

| Size (sq. ft.) | Age (years) | Price ($1000s) |
|----------------|-------------|----------------|
| 1500           | 5           | 300            |
| 2000           | 10          | 350            |
| 1200           | 2           | 250            |
| 1800           | 8           | 320            |
| 2200           | 15          | 380            |
| 1600           | 3           | 290            |
| 2500           | 20          | 400            |
| 1900           | 12          | 340            |
| 1700           | 7           | 310            |

Compute by hand the error for linear regression model 
$$(Price) y  = 0.1\cdot Size + 2\cdot Age +100$$



## Exercise 7

The file `car_efficiency.csv` contains the MPGs and the weights of different cars.

1.  Plot as a scatter plot the data
2.  Find the linear regression model, this means, let x=MPG of the ith car and y=weight of the ith car.
$$y_i \approx m\cdot x_i + b$$
1.  Find the quadratic regression model, this means.
$$y_i \approx m_2\cdot x_i^2+ m_1\cdot x_i + b$$
1. Plot the scatter plot of the data with the regression line.
2. Estimate the weight of a car if the mileage is 32 MPG

To read the file use the following command:

```python
import pandas as pd
df = pd.read_csv("car_efficiency.csv") # read the csv file and put the data into the data frame df
values = df.values # puit all the values of the data frame into an array values, where the first column is x and the second colum is y
x = values[:,0]
y = values[:,1]
```

## Exercise 8
Find a data set with at least 2 feature (this means x has to be of dimension 2 or more) and find the linear regression line to predict the label y.