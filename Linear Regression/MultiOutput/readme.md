# MultiOutput Linear Regression

This folder contains the Linear Regression ML model used to estimate quadrotor parameters from simulated flight data.

- **LinearRegressionTrain.py** – Trains the decision tree model  
- **LinearRegressionTest.py** – Tests and evaluates model performance

## Results (so far)

| Parameter | Mean Training Error (%) | Mean Test Error (%) |
|------------|-------------------:|---------------:|
| Thrust Coeff | 136.38 | 119.67 |
| Drag Coeff | 156.24 | 105.69 |
| X-Inertia | 85.43 | 94.69 |
| Y-Inertia | 80.52 | 101.94 |
| Z-Inertia | 82.15 | 98.38 |

The plots below show the predicted vs. actual parameter values (left) and the percent error (right) for the test data.

![Decision Tree Results](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/blob/20660df14bb6c238dfdaefc8432d7d8bc28bf3e3/Linear%20Regression/MultiOutput/Figures/MultiOutput%20Linear%20Regression%20Results.png)


