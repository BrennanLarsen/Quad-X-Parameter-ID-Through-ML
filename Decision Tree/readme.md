# Decision Tree

This folder contains the machine learning model used to estimate quadrotor parameters from simulated flight data.

- **DecisionTree_Train.py** – Trains the decision tree model  
- **DecisionTree_Test.py** – Tests and evaluates model performance

## Results

**Training Set Mean Percent Error**
| Parameter | Error |
|------------|--------|
| Thrust Coeff | 0.00% |
| Drag Coeff | 22.27% |
| X-Inertia | 0.00% |
| Y-Inertia | 0.00% |
| Z-Inertia | 21.75% |

**Test Set Mean Percent Error**
| Parameter | Error |
|------------|--------|
| Thrust Coeff | 29.71% |
| Drag Coeff | 38.34% |
| X-Inertia | 31.03% |
| Y-Inertia | 20.17% |
| Z-Inertia | 36.12% |

## Visualization
The plots below show the predicted vs. actual parameter values (left) and the percent error (right).

![Decision Tree Results](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/blob/25db1e27de723fc43bcdef844c3ae299249fb109/Decision%20Tree/Figures/Decision%20Tree%20Results.png)

