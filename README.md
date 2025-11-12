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

- **[Decision Tree](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/tree/main/Decision%20Tree)**  
  Baseline machine learning models for parameter prediction.  
  - **MultiOutput** – Predicts all five parameters (thrust, drag, and inertias) simultaneously.  
  - **SingleOutput** – Predicts one parameter at a time.
 
- **[Linear Regression](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/tree/main/Linear%20Regression)**  
  Not expected to preform well but a simple model worth checking.  
  - **MultiOutput** – Predicts all five parameters (thrust, drag, and inertias) simultaneously.  
  - **SingleOutput** – Predicts one parameter at a time.  

## Approach
1. **Data Generation**  
   - Training data: Piecewise design of experiments (DOE).  
   - Testing data: Latin Hypercube Sampling (LHS) for randomized, well-distributed parameter combinations.

2. **Exploratory Data Analysis**  
   - Examine distributions, correlations, and feature importance.  
   - Identify features most informative for predicting parameters.

3. **Modeling**  
   - Baseline: Decision tree for interpretable, nonlinear predictions.  
   - Future: Neural networks or online learning models may be explored for improved accuracy.
  

## Results Summerized

| Parameter     | <div align="center">Decision Tree<br><div align="center">MultiOutput<br>Mean Test Error (%)</div> | <div align="center">Decision Tree<br><div align="center">SingleOutput<br>Mean Test Error (%)</div> | <div align="center">Linear Regression<br><div align="center">MultiOutput<br>Mean Test Error (%)</div> | <div align="center">Linear Regression<br><div align="center">SingleOutput<br>Mean Test Error (%)</div> |
|---------------|:------------------------------------------------------------:|:--------------------------------------------------------------:|:------------------------------------------------------------:|:--------------------------------------------------------------:|
| Thrust Coeff  | 29.71 | 29.71 | 119.67 | 119.67 |
| Drag Coeff    | 38.34 | 42.81 | 105.69 | 105.69 |
| X-Inertia     | 31.03 | 31.03 | 94.69 | 94.69 |
| Y-Inertia     | 20.17 | 20.17 | 101.94 | 101.94 |
| Z-Inertia     | 36.12 | 43.63 | 98.38 | 98.38 |





