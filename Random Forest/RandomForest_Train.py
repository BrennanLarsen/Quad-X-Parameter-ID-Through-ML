import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))


# ================================================== #
#     Hyperparameters
# ================================================== #
model_path = Path(__file__).parent / "RandomForest_Model.pkl"

# Data file paths
data_path = r"TRAIN_DATA_FILE_PATH_HERE.xlsx"
test_data_path = r"TEST_DATA_FILE_PATH_HERE.xlsx"

n_estimators = 100
max_depth = 12
min_samples_split = 6
min_samples_leaf = 3
max_features = 'sqrt'   # usually 'sqrt', 'log2', or float
n_folds = 3

hyperparameters = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'max_features': max_features,
    'n_folds': n_folds
}

# Parameter column
output_cols = ['c_T','c_RD','I_x','I_y','I_z']

# normalize
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

# feature matrix
X_train = df[feature_cols].values

# Normalize output data
df_outputs_norm, output_norm_params = normalize_data(df_outputs, normalize_output_cols)
Y_train = df_outputs_norm.values


# ================================================== #
#    Cross validation
# ================================================== #
output_names = ["Thrust Coeff", "Drag Coeff", "X-Inertia", "Y-Inertia", "Z-Inertia"]
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=3)
cv_results = {name: [] for name in output_names}

for train_idx, val_idx in kfold.split(X_train):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    Y_fold_train, Y_fold_val = Y_train[train_idx], Y_train[val_idx]
    
    fold_model = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion='squared_error',
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    
    fold_model.fit(X_fold_train, Y_fold_train)
    Y_fold_pred = fold_model.predict(X_fold_val)
    
    # denormalize outputs
    Y_val_denorm = denormalize_data(Y_fold_val, output_norm_params, normalize_output_cols)
    Y_pred_denorm = denormalize_data(Y_fold_pred, output_norm_params, normalize_output_cols)
    
    # output errors
    for i, name in enumerate(output_names):
        percent_error = ((Y_pred_denorm[:, i] - Y_val_denorm[:, i]) / Y_val_denorm[:, i]) * 100
        # handle divide by zero
        percent_error = np.nan_to_num(percent_error, nan=0.0, posinf=0.0, neginf=0.0)
        # Store
        cv_results[name].extend(percent_error.tolist())


# ================================================== #
#    Train final model
# ================================================== #
model = RandomForestRegressor(
    n_estimators=n_estimators,
    criterion='squared_error',
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
    random_state=3,
    n_jobs=-1
)

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
    'output_norm_params': output_norm_params,
    'normalize_output_cols': normalize_output_cols,
    'feature_cols': feature_cols,
    'cv_results': cv_results
}

# Save model
joblib.dump(model_data, model_path)


# ================================================== #
#    Test model
# ================================================== #
from RandomForest_Test import evaluate_model
evaluate_model(model_path, test_data_path, plot_results=True)