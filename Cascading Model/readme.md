# Cascading Model

This cascading model has three layers that progressively refine predictions: 

1. Linear Regression - Initial linear predictions
2. Decision Trees - Two parallel trees refining aerodynamic and inertial outputs separately
3. Random Forest - Final layer combining all previous predictions

Each layer's outputs become additional features for the next layer.


## Results

| Parameter | Mean Training Error (%) | Mean Test Error (%) |
|------------|-------------------:|---------------:|
| Thrust Coeff | 6.83 | 9.03 |
| Drag Coeff | 13.97 | 16.37 |
| X-Inertia | 5.69 | 8.51 |
| Y-Inertia | 5.71 | 8.09 |
| Z-Inertia | 14.50 | 15.01 |


The plots below show the predicted vs. actual parameter values (left) and the percent error (right) for the test data.

![Cascading Model Results](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/blob/cbd4f59a67737e68d1d849fee6cee15159659879/Cascading%20Model/Figures/Cascading%20Model%20Results.png)


