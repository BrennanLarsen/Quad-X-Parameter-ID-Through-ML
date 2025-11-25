import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPRegressor
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))


# ================================================== #
#     Hyperparameters
# ================================================== #
model_path = Path(__file__).parent / "NeuralNet_Model.pkl"

# Data file paths
data_path = r"TRAIN_DATA_FILE_PATH_HERE.xlsx"
test_data_path = r"TEST_DATA_FILE_PATH_HERE.xlsx"

hidden_layer_sizes = (396, 39)
activation = 'tanh'
learning_rate_init = 0.001
epochs = 5000

hyperparameters = {
    'hidden_layer_sizes': hidden_layer_sizes,
    'activation': activation,
    'learning_rate_init': learning_rate_init,
    'epochs': epochs
}

output_cols = ['c_T','c_RD','I_x','I_y','I_z']

# Columns to normalize
normalize_output_cols = ['c_T','c_RD','I_x','I_y','I_z']


# ================================================== #
#    Helper Functions
# ================================================== #
def normalize_data(data, columns):      # min-max normalization, save params
    norm_params = {}
    data_normalized = data.copy()
    
    for col in columns:
        min_val = data[col].min()
        max_val = data[col].max()

        # Avoid divide by zero
        if max_val - min_val == 0:
            data_normalized[col] = 0
            norm_params[col] = {'min': min_val, 'max': max_val, 'range': 1}
        else:
            data_normalized[col] = (data[col] - min_val) / (max_val - min_val)
            norm_params[col] = {'min': min_val, 'max': max_val, 'range': max_val - min_val}
    
    return data_normalized, norm_params


def denormalize_data(data_normalized, norm_params, columns):       # reverse norm using stored params
    data_denormalized = data_normalized.copy()
    
    for i, col in enumerate(columns):
        min_val = norm_params[col]['min']
        range_val = norm_params[col]['range']
        data_denormalized[:, i] = data_normalized[:, i] * range_val + min_val
    
    return data_denormalized


# ================================================== #
#    Load and prep data
# ================================================== #
df = pd.read_excel(data_path, engine='openpyxl')

# Separate parameters from features
df_outputs = df[output_cols].copy()

param_cols = ["Run", "c_T", "c_RD", "I_x", "I_y", "I_z", "m", "l", "angle_motor1_2"]
feature_cols = [col for col in df.columns if col not in param_cols]

# Extract features
X_train = df[feature_cols].values

# Normalize output data
df_outputs_norm, output_norm_params = normalize_data(df_outputs, normalize_output_cols)
Y_train = df_outputs_norm.values


# ================================================== #
#    Train final model
# ================================================== #
model = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    solver='adam',
    learning_rate_init=learning_rate_init,
    max_iter=epochs,
    random_state=3
)

model.fit(X_train, Y_train)

# run on training data
output_names = ["Thrust Coeff", "Drag Coeff", "X-Inertia", "Y-Inertia", "Z-Inertia"]
Y_train_pred_norm = model.predict(X_train)
Y_train_denorm = denormalize_data(Y_train, output_norm_params, normalize_output_cols)
Y_train_pred_denorm = denormalize_data(Y_train_pred_norm, output_norm_params, normalize_output_cols)

print("\nTRAINING SET MEAN PERCENT ERROR:")
print("="*50)
for i, name in enumerate(output_names):
    errors = Y_train_pred_denorm[:, i] - Y_train_denorm[:, i]
    percent_errors = (errors / Y_train_denorm[:, i]) * 100
    mean_percent_error = np.mean(np.abs(percent_errors))
    print(f"{name:<25} {mean_percent_error:>6.2f}%")


# ================================================== #
#    Save model
# ================================================== #
model_data = {
    'model': model,
    'hyperparameters': hyperparameters,
    'output_norm_params': output_norm_params,
    'normalize_output_cols': normalize_output_cols,
    'feature_cols': feature_cols
}

# Save model
joblib.dump(model_data, model_path)


# ================================================== #
#    Test model
# ================================================== #
from NeuralNet_Test import evaluate_model
evaluate_model(model_path, test_data_path, plot_results=True)