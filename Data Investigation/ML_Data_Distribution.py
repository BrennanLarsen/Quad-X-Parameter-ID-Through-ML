import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, norm
from pathlib import Path

# ================================================== #
#    Data
# ================================================== #
data_path = Path(__file__).parent / "DATA_PATH_HERE.xlsx"
df = pd.read_excel(data_path)

# ================================================== #
#    Group variables
# ================================================== #
groups = {
    "Linear Values": ['x','y','z','dx','dy','dz','ddx','ddy','ddz'],
    "Angular Values": ['phi','theta','psi','p','q','r','dp','dq','dr'],
    "Motor Speeds": ['omega_1','omega_2','omega_3','omega_4'],
    "Physical Parameters": ['m', 'c_T','c_RD','I_x','I_y','I_z']
}

custom_labels = {
    "x": "X-Position", "y": "Y-Position", "z": "Z-Position",
    "dx": "X-Velocity", "dy": "Y-Velocity", "dz": "Z-Velocity",
    "ddx": "X-Accel", "ddy": "Y-Accel", "ddz": "Z-Accel",
    "phi": "Roll Angle", "theta": "Pitch Angle", "psi": "Yaw Angle",
    "p": "Roll Rate", "q": "Pitch Rate", "r": "Yaw Rate",
    "dp": "Roll Accel", "dq": "Pitch Accel", "dr": "Yaw Accel",
    "omega_1": "Motor 1 Speed", "omega_2": "Motor 2 Speed",
    "omega_3": "Motor 3 Speed", "omega_4": "Motor 4 Speed",
    "m": "Mass", "c_T": "Thrust Coeff", "c_RD": "Drag Coeff",
    "I_x": "Inertia X", "I_y": "Inertia Y", "I_z": "Inertia Z"
}

# ================================================== #
#    Plot distributions
# ================================================== #
def plot_group_distributions(df, group_name, cols, labels=None, bins=20, ncols=3):
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(4*ncols, 3*nrows))
    
    for i, col in enumerate(cols, 1):
        values = df[col].to_numpy()
        mean, std = np.mean(values), np.std(values)
        skw = skew(values)
        
        plt.subplot(nrows, ncols, i)
        plt.hist(values, bins=bins, density=True)
        
        # Normal fit
        x_fit = np.linspace(min(values), max(values))
        plt.plot(x_fit, norm.pdf(x_fit, mean, std), 'r')
        
        title = labels.get(col, col) if labels else col
        plt.title(f"{title}\nμ={mean:.2f}, σ={std:.2f}, skew={skw:.2f}", fontsize=9)
        plt.grid(axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.suptitle(f"Distribution of {group_name}", fontsize=14)
    plt.show()

group_columns = {
    "Linear Movement": 3,
    "Angular": 3,
    "Motor Speeds": 2,
    "Physical Parameters": 2
}

for group_name, cols in groups.items():
    ncols = group_columns.get(group_name, 3)  
    plot_group_distributions(df, group_name, cols, labels=custom_labels, ncols=ncols)
