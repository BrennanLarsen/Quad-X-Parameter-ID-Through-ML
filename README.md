# Quadrotor(X) Parameter Identification Through Machine Learning


## Overview
Accurately identifying a quadrotor’s physical parameters (moments of inertia (X, Y, Z), thrust, and drag) is crucial for precise control and stable flight. Traditional methods are labor-intensive and sensitive to hardware changes. This project explores using machine learning to predict these parameters directly from flight data without requiring full analytical modeling.


## Repository Structure

- **[Quadrotor-X Dynamic Simulation](https://github.com/BrennanLarsen/Quadrotor-X-Dynamic-Simulation)** (separate repo)  
  Physics-based simulation used to generate training and testing flight data.  

- **[Data](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/tree/main/Data)**  
  Contains all the datasets.  

- **[Data Investigation](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/tree/main/Data%20Investigation)**  
  Programs for analyzing distributions, correlations, and feature importance for data exploration.


Machine learning models that were investigated:
- **[Decision Tree (Multi and Single Output)](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/tree/main/Decision%20Tree)**  
- **[Linear Regression (Multi and Single Output)](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/tree/main/Linear%20Regression)**
- **[Random Forest (Multi Output)](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/tree/main/Random%20Forest)**
- **[Cascading Model](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/tree/main/Cascading%20Model)**  

  
## Results Summerized

| Parameter | SingleOutput<br>Linear&nbsp;Regression<br>Train&nbsp;(%)Error | SingleOutput<br>Linear&nbsp;Regression<br>Test&nbsp;(%)Error | MultiOutput<br>Linear&nbsp;Regression<br>Train&nbsp;(%)Error | MultiOutput<br>Linear&nbsp;Regression<br>Test&nbsp;(%)Error | SingleOutput<br>Decision&nbsp;Tree<br>Train&nbsp;(%)Error | SingleOutput<br>Decision&nbsp;Tree<br>Test&nbsp;(%)Error | MultiOutput<br>Decision&nbsp;Tree<br>Train&nbsp;(%)Error | MultiOutput<br>Decision&nbsp;Tree<br>Test&nbsp;(%)Error | MultiOutput<br>Random&nbsp;Forest<br>Train&nbsp;(%)Error | MultiOutput<br>Random&nbsp;Forest<br>Test&nbsp;(%)Error | MultiOutput<br>Cascading&nbsp;Model<br>Train&nbsp;(%)Error | MultiOutput<br>Cascading&nbsp;Model<br>Test&nbsp;(%)Error | MultiOutput<br>Neural&nbsp;Network<br>Train&nbsp;(%)Error | MultiOutput<br>Neural&nbsp;Network<br>Test&nbsp;(%)Error |
|:------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Thrust&nbsp;Coeff | 136.38 | 119.67 | 136.38 | 119.67 | 0.00 | 29.71 | 0.00 | 11.10 | 8.85 | 11.19 | 6.83 | 9.03 | 121.84 | 87.42 |
| Drag&nbsp;Coeff   | 156.24 | 105.69 | 156.24 | 105.69 | 0.00 | 42.81 | 0.00 | 18.57 | 19.52 | 19.44 | 13.97 | 16.37 | 109.08 | 84.98 |
| X-Inertia    | 85.43 | 94.69 | 85.43 | 94.69 | 0.00 | 31.03 | 0.00 | 10.22 | 8.09 | 11.70 | 5.69 | 8.51 | 92.81 | 80.43 |
| Y-Inertia    | 80.52 | 101.94 | 80.52 | 101.94 | 0.00 | 20.17 | 0.00 | 10.24 | 8.09 | 12.36 | 5.71 | 8.09 | 102.50 | 81.02 |
| Z-Inertia    | 82.15 | 98.38 | 82.15 | 98.38 | 0.00 | 43.63 | 0.00 | 18.61 | 20.95 | 19.03 | 14.50 | 15.01 | 83.00 | 57.89 |




| Parameter | Mean Training Error (%) | Mean Test Error (%) |
|------------|-------------------:|---------------:|
| Thrust Coeff | 121.84 | 87.42 |
| Drag Coeff | 109.08 | 84.98 |
| X-Inertia | 92.81 | 80.43 |
| Y-Inertia | 102.50 | 81.02 |
| Z-Inertia | 83.00 | 57.89 |








