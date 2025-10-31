# DecisionTree_Train.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))


# ================================================== #
#     Hyperparameters
# ================================================== #
model_path = Path(__file__).parent / "DecisionTree_Model.pkl"

# Data file paths
data_path = r"TRAINING_DATA_PATH_HERE.xlsx"
test_data_path = r"TESTING_DATA_PATH_HERE.xlsx"

max_depth = 13
min_samples_split = 4
min_samples_leaf = 10
sequence_length = 297   # samples per flight sequence
n_folds = 6             # K-fold CV splits

hyperparameters = {
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'sequence_length': sequence_length,
    'n_folds': n_folds
}

input_cols = ['t','x','y','z','dx','dy','dz','ddx','ddy','ddz','phi','theta','psi','p','q','r','dp','dq','dr',
              'omega_1','omega_2','omega_3','omega_4','m']
output_cols = ['c_T','c_RD','I_x','I_y','I_z']

# Columns to normalize
normalize_input_cols = ['t','x','y','z','dx','dy','dz','ddx','ddy','ddz','phi','theta','psi','p','q','r','dp','dq','dr',
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
        data_denormalized[:, i] = data_normalized[:, i] * range_val + min_val
    
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
    parameters = df_outputs_norm.values[start_idx, :]  # Same params for whole sequence
    
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
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=3)
cv_results = {name: [] for name in output_names}

# perform k fold cross validation and record % error
for train_idx, val_idx in kfold.split(X_train):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    Y_fold_train, Y_fold_val = Y_train[train_idx], Y_train[val_idx]
    
    base_model = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    fold_model = MultiOutputRegressor(base_model)
    fold_model.fit(X_fold_train, Y_fold_train)
    Y_fold_pred = fold_model.predict(X_fold_val)
    
    # denormalize outputs
    Y_val_denorm = denormalize_data(Y_fold_val, output_norm_params, normalize_output_cols)
    Y_pred_denorm = denormalize_data(Y_fold_pred, output_norm_params, normalize_output_cols)
    
    # Compute % error for each output
    for i, name in enumerate(output_names):
        # percent error for each sample
        percent_error = ((Y_pred_denorm[:, i] - Y_val_denorm[:, i]) / Y_val_denorm[:, i]) * 100
        # handle divide by zero
        percent_error = np.nan_to_num(percent_error, nan=0.0, posinf=0.0, neginf=0.0)
        # Store errors
        cv_results[name].extend(percent_error.tolist())


# ================================================== #
#    Train final model
# ================================================== #
base_model = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=3
)

# MultiOutputRegressor for predicting multiple parameters
model = MultiOutputRegressor(base_model)
model.fit(X_train, Y_train)

# run on training data
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
from DecisionTree_Test import evaluate_model
evaluate_model(model_path, test_data_path, plot_results=True)