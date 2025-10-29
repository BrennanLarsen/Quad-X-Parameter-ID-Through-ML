# Quadrotor(X) Parameter Identification Through Machine Learning

## Overview
Accurately identifying a quadrotor’s physical parameters (moments of inertia (X, Y, Z), thrust, and drag) is crucial for precise control and stable flight. Traditional methods are labor-intensive and sensitive to hardware changes. This project explores using **machine learning** to predict these parameters directly from flight data without requiring full analytical modeling.

## Repository Structure

- **[Quadrotor-X Dynamic Simulation](https://github.com/BrennanLarsen/Quadrotor-X-Dynamic-Simulation)** (separate repo)  
  Physics-based simulation used to generate training and testing flight data.  

- **[Data](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/tree/main/Data)**  
  Contains all the datasets.  

- **[Data Investigation](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/tree/main/Data%20Investigation)**  
  Contains all the programs for analyzing distributions, correlations, feature importance, etc. for investigating the data.

- **[Decision Tree](https://github.com/BrennanLarsen/Quad-X-Parameter-ID-Through-ML/tree/main/Decision%20Tree)**  
  Baseline machine learning models for parameter prediction.  
  

## Approach
1. **Data Generation**  
   - Training data: Piecewise design of experiments (DOE).  
   - Testing data: Latin Hypercube Sampling (LHS) for randomized, well-distributed parameter combinations.

2. **Exploratory Data Analysis**  
   - Examine distributions, correlations, feature importance and engineering.  
   - Identify features most informative for predicting parameters.

3. **Modeling**  
   - Baseline: Decision tree for interpretable, nonlinear predictions.  
   - Future: Neural networks or online learning models may be explored for improved accuracy.
