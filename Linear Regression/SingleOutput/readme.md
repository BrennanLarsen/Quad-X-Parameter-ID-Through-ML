# SingleOutput Linear Regression

This folder contains the Linear Regression ML model used to estimate quadrotor parameters from simulated flight data for a single output (parameter) at a time.

- **SingleLinearRegressionTrain.py** – Trains the Linear Regression model (single output) 
- **SingleLinearRegressionTest.py** – Tests and evaluates model performance (single output)

## Results (so far)

| Parameter | Mean Training Error (%) | Mean Test Error (%) |
|------------|-------------------:|---------------:|
| Thrust Coeff | 136.38 | 119.6 |
| Drag Coeff | 156.24 | 105.69 |
| X-Inertia | 85.43 | 94.69 |
| Y-Inertia | 80.52 | 101.94 |
| Z-Inertia | 82.15 | 98.38 |

The plots below show the predicted vs. actual parameter values (left) and the percent error (right) for the test data (single output).

![Thrust Coeff Model Results]()

![Rotor Drag Coeff Model Results]()

![X-Inertia Model Results]()

![Y-Inertia Model Results]()

![Z-Inertia Model Results]()

