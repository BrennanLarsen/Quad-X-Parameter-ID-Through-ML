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

| Parameter     | <div align="center">Decision Tree<br><div align="center">MultiOutput<br>Mean Test Error (%)</div> | <div align="center">Decision Tree<br><div align="center">SingleOutput<br>Mean Test Error (%)</div> | <div align="center">Linear Regression<br><div align="center">MultiOutput<br>Mean Test Error (%)</div> | <div align="center">Linear Regression<br><div align="center">SingleOutput<br>Mean Test Error (%)</div> |
|---------------|:------------------------------------------------------------:|:--------------------------------------------------------------:|:------------------------------------------------------------:|:--------------------------------------------------------------:|
| Thrust Coeff  | 29.71 | 29.71 | 119.67 | 119.67 |
| Drag Coeff    | 38.34 | 42.81 | 105.69 | 105.69 |
| X-Inertia     | 31.03 | 31.03 | 94.69 | 94.69 |
| Y-Inertia     | 20.17 | 20.17 | 101.94 | 101.94 |
| Z-Inertia     | 36.12 | 43.63 | 98.38 | 98.38 |









