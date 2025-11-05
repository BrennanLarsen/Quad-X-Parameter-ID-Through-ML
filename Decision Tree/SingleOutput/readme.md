# SingleOutput Decision Tree

This folder contains the decision tree ML model used to estimate quadrotor parameters from simulated flight data for a single output (parameter) at a time.

- **SingleDecisionTree_Train.py** – Trains the decision tree model (single output) 
- **SingleDecisionTree_Test.py** – Tests and evaluates model performance (single output)

## Results (so far)

| Parameter | Mean Training Error (%) | Mean Test Error (%) |
|------------|-------------------:|---------------:|
| Thrust Coeff | 0.00 | 29.71 |
| Drag Coeff | 0.00 | 42.81 |
| X-Inertia | 0.00 | 31.03 |
| Y-Inertia | 0.00 | 20.17 |
| Z-Inertia | 0.00 | 43.63 |

The plots below show the predicted vs. actual parameter values (left) and the percent error (right) for the test data (single output).

![Thrust Coeff Model Results](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/blob/d74657368c75504fde19cc1304cbc9bd68b9f4eb/Decision%20Tree/SingleOutput/Figures/Thrust%20Coeff%20Model%20Results.png)

![Rotor Drag Coeff Model Results](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/blob/d74657368c75504fde19cc1304cbc9bd68b9f4eb/Decision%20Tree/SingleOutput/Figures/Rotor%20Drag%20Coeff%20Model%20Results.png)

![X-Inertia Model Results](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/blob/d74657368c75504fde19cc1304cbc9bd68b9f4eb/Decision%20Tree/SingleOutput/Figures/X-Inertia%20Model%20Results.png)

![Y-Inertia Model Results](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/blob/d74657368c75504fde19cc1304cbc9bd68b9f4eb/Decision%20Tree/SingleOutput/Figures/Y-Inertia%20Model%20Results.png)

![Z-Inertia Model Results](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/blob/d74657368c75504fde19cc1304cbc9bd68b9f4eb/Decision%20Tree/SingleOutput/Figures/Z-Inertia%20Model%20Results.png)
