# SingleLinearRegressionTrain.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))


# ================================================== #
#     Hyperparameters
# ================================================== #
model_path = Path(__file__).parent / "LinearRegression_Model.pkl"

# Data file paths
data_path = r"TRAIN_DATA_PATH_HERE.xlsx"
test_data_path = r"TEST_DATA_PATH_HERE.xlsx"


# feature to output
# 0: c_T (Thrust Coeff)
# 1: c_RD (Drag Coeff)
# 2: I_x (X-Inertia)
# 3: I_y (Y-Inertia)
# 4: I_z (Z-Inertia)
       
TARGET_FEATURE_INDEX = 4  
sequence_length = 296   # samples per flight sequence
n_folds = 6             # K-fold CV splits

hyperparameters = {
    'sequence_length': sequence_length,
    'n_folds': n_folds,
    'target_feature_index': TARGET_FEATURE_INDEX
}

input_cols = ['t','x','y','z','dx','dy','dz','ddx','ddy','ddz','phi','theta','psi','p','q','r','dp','dq','dr',
              'omega_1','omega_2','omega_3','omega_4','m']
output_cols = ['c_T','c_RD','I_x','I_y','I_z']

# Columns to normalize
normalize_input_cols = ['x','y','z','dx','dy','dz','ddx','ddy','ddz','phi','theta','psi','p','q','r','dp','dq','dr',
                        'omega_1','omega_2','omega_3','omega_4','m']
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
        # Handle 2D arrays and 1D arrays
        if data_denormalized.ndim == 2:
            data_denormalized[:, i] = data_normalized[:, i] * range_val + min_val
        else:
            data_denormalized[i] = data_normalized[i] * range_val + min_val
    
    return data_denormalized


def feature_engineering(sequence):
    mean_vals = np.mean(sequence, axis=0)
    std_vals = np.std(sequence, axis=0)
    min_vals = np.min(sequence, axis=0)
    max_vals = np.max(sequence, axis=0)
    first_vals = sequence[0, :]
    last_vals = sequence[-1, :]
    range_vals = max_vals - min_vals
    median_vals = np.median(sequence, axis=0)
    
    # all features into one vector
    features = np.concatenate([
        mean_vals, std_vals, min_vals, max_vals, 
        first_vals, last_vals, range_vals, median_vals
    ])
    return features


# ================================================== #
#    Load and prep data
# ================================================== #
df = pd.read_excel(data_path, engine='openpyxl')
df_inputs = df[input_cols].copy()
df_outputs = df[output_cols].copy()

# Normalize input/output data
df_inputs_norm, input_norm_params = normalize_data(df_inputs, normalize_input_cols)
df_outputs_norm, output_norm_params = normalize_data(df_outputs, normalize_output_cols)

# Split data into sequences
num_quadrotors = len(df) // sequence_length
X_train, Y_train = [], []

for i in range(num_quadrotors):
    start_idx = i * sequence_length
    end_idx = start_idx + sequence_length
    flight_inputs = df_inputs_norm.values[start_idx:end_idx, :]
    parameters = df_outputs_norm.values[start_idx, TARGET_FEATURE_INDEX]  # Extract single feature
    
    # Extract features for this sequence
    features = feature_engineering(flight_inputs)
    X_train.append(features)
    Y_train.append(parameters)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


# ================================================== #
#    Cross validation
# ================================================== #
output_names = ["Thrust Coeff", "Drag Coeff", "X-Inertia", "Y-Inertia", "Z-Inertia"]
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
cv_results = []

# perform k fold cross validation and record % error
for train_idx, val_idx in kfold.split(X_train):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    Y_fold_train, Y_fold_val = Y_train[train_idx], Y_train[val_idx]
    
    fold_model = LinearRegression()
    
    fold_model.fit(X_fold_train, Y_fold_train)
    Y_fold_pred = fold_model.predict(X_fold_val)
    
    # denormalize outputs
    Y_val_denorm = Y_fold_val * output_norm_params[output_cols[TARGET_FEATURE_INDEX]]['range'] + \
                   output_norm_params[output_cols[TARGET_FEATURE_INDEX]]['min']
    Y_pred_denorm = Y_fold_pred * output_norm_params[output_cols[TARGET_FEATURE_INDEX]]['range'] + \
                    output_norm_params[output_cols[TARGET_FEATURE_INDEX]]['min']
    
    # Compute % error for each output
    percent_error = ((Y_pred_denorm - Y_val_denorm) / Y_val_denorm) * 100
    # handle divide by zero
    percent_error = np.nan_to_num(percent_error, nan=0.0, posinf=0.0, neginf=0.0)
    # Store errors
    cv_results.extend(percent_error.tolist())


# ================================================== #
#    Train final model
# ================================================== #
base_model = LinearRegression()

model = base_model
model.fit(X_train, Y_train)

# run on training data
Y_train_pred_norm = model.predict(X_train)
Y_train_denorm = Y_train * output_norm_params[output_cols[TARGET_FEATURE_INDEX]]['range'] + \
                 output_norm_params[output_cols[TARGET_FEATURE_INDEX]]['min']
Y_train_pred_denorm = Y_train_pred_norm * output_norm_params[output_cols[TARGET_FEATURE_INDEX]]['range'] + \
                      output_norm_params[output_cols[TARGET_FEATURE_INDEX]]['min']

print("\nTRAINING SET MEAN PERCENT ERROR:")
print("="*50)
errors = Y_train_pred_denorm - Y_train_denorm
percent_errors = (errors / Y_train_denorm) * 100
mean_percent_error = np.mean(np.abs(percent_errors))
print(f"{output_names[TARGET_FEATURE_INDEX]:<25} {mean_percent_error:>6.2f}%")


# ================================================== #
#    Save model
# ================================================== #
model_data = {
    'model': model,
    'hyperparameters': hyperparameters,
    'input_norm_params': input_norm_params,
    'output_norm_params': output_norm_params,
    'normalize_input_cols': normalize_input_cols,
    'normalize_output_cols': normalize_output_cols,
    'cv_results': cv_results
}

# Save model
joblib.dump(model_data, model_path)


# ================================================== #
#    Test model
# ================================================== #
from SingleLinearRegressionTest import evaluate_model
evaluate_model(model_path, test_data_path, plot_results=True)