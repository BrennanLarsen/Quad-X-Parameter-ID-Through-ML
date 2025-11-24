import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
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
model_path = Path(__file__).parent / "Cascading_Model.pkl"

# Data file paths
data_path = r"TRAIN_DATA_FILE_PATH_HERE.xlsx"
test_data_path = r"TEST_DATA_FILE_PATH_HERE.xlsx"

n_folds = 3  

# Layer 1 (LR)
ridge_alpha = 1.0

# Layer 2 (DT)
dt_max_depth = 12
dt_min_samples_split = 5
dt_min_samples_leaf = 2

# Layer 3 (RF)
rf_n_estimators = 150
rf_max_depth = 15
rf_min_samples_split = 4
rf_min_samples_leaf = 2
rf_max_features = 'sqrt'

output_cols = ['c_T', 'c_RD', 'I_x', 'I_y', 'I_z']


# ================================================== #
#    Helper Functions
# ================================================== #
def normalize_data(data, columns):
    norm_params = {}
    data_normalized = data.copy()
    
    for col in columns:
        min_val = data[col].min()
        max_val = data[col].max()

        if max_val - min_val == 0:
            data_normalized[col] = 0
            norm_params[col] = {'min': min_val, 'max': max_val, 'range': 1}
        else:
            data_normalized[col] = (data[col] - min_val) / (max_val - min_val)
            norm_params[col] = {'min': min_val, 'max': max_val, 'range': max_val - min_val}
    
    return data_normalized, norm_params


def denormalize_data(data_normalized, norm_params, columns):
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
df_outputs = df[output_cols].copy()

# parameters
param_cols = ["Run", "c_T", "c_RD", "I_x", "I_y", "I_z", "m"]
feature_cols = [col for col in df.columns if col not in param_cols]

# feature matrix
X_train_base = df[feature_cols].values

# Normalize output 
df_outputs_norm, output_norm_params = normalize_data(df_outputs, output_cols)
Y_train = df_outputs_norm.values


# ================================================== #
#    Cross validation
# ================================================== #
output_names = ["Thrust Coeff", "Drag Coeff", "X-Inertia", "Y-Inertia", "Z-Inertia"]
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=3)
cv_results = {name: [] for name in output_names}

for fold_num, (train_idx, val_idx) in enumerate(kfold.split(X_train_base), 1):
    
    X_fold_train = X_train_base[train_idx]
    X_fold_val = X_train_base[val_idx]
    Y_fold_train = Y_train[train_idx]
    Y_fold_val = Y_train[val_idx]
    
    # Layer 1 (LR)
    layer1 = Ridge(alpha=ridge_alpha, random_state=42)
    layer1.fit(X_fold_train, Y_fold_train)
    layer1_train = layer1.predict(X_fold_train)
    layer1_val = layer1.predict(X_fold_val)
    
    # add Layer 1 predictions as new features
    X_fold_train = np.column_stack([X_fold_train, layer1_train])
    X_fold_val = np.column_stack([X_fold_val, layer1_val])
    
    # Layer 2a (DT for c_T, c_RD)
    layer2a = MultiOutputRegressor(
        DecisionTreeRegressor(
            max_depth=dt_max_depth,
            min_samples_split=dt_min_samples_split,
            min_samples_leaf=dt_min_samples_leaf,
            random_state=42
        )
    )
    layer2a.fit(X_fold_train, Y_fold_train[:, [0, 1]])
    layer2a_train = layer2a.predict(X_fold_train)
    layer2a_val = layer2a.predict(X_fold_val)
    
    # Layer 2b (DT for I_x, I_y, I_z)
    layer2b = MultiOutputRegressor(
        DecisionTreeRegressor(
            max_depth=dt_max_depth,
            min_samples_split=dt_min_samples_split,
            min_samples_leaf=dt_min_samples_leaf,
            random_state=42
        )
    )
    layer2b.fit(X_fold_train, Y_fold_train[:, [2, 3, 4]])
    layer2b_train = layer2b.predict(X_fold_train)
    layer2b_val = layer2b.predict(X_fold_val)
    
    # add Layer 2 predictions as new features
    X_fold_train = np.column_stack([X_fold_train, layer2a_train, layer2b_train])
    X_fold_val = np.column_stack([X_fold_val, layer2a_val, layer2b_val])
    
    # Layer 3 (RF)
    layer3 = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
        min_samples_leaf=rf_min_samples_leaf,
        max_features=rf_max_features,
        random_state=42,
        n_jobs=-1
    )
    layer3.fit(X_fold_train, Y_fold_train)
    Y_fold_pred = layer3.predict(X_fold_val)
    
    # Denormalize and errors
    Y_val_denorm = denormalize_data(Y_fold_val, output_norm_params, output_cols)
    Y_pred_denorm = denormalize_data(Y_fold_pred, output_norm_params, output_cols)
    
    for i, name in enumerate(output_names):
        percent_error = ((Y_pred_denorm[:, i] - Y_val_denorm[:, i]) / Y_val_denorm[:, i]) * 100
        percent_error = np.nan_to_num(percent_error, nan=0.0, posinf=0.0, neginf=0.0)
        cv_results[name].extend(percent_error.tolist())


# ================================================== #
#    Train final model
# ================================================== #
# Layer 1
layer1_model = Ridge(alpha=ridge_alpha, random_state=42)
layer1_model.fit(X_train_base, Y_train)
layer1_pred = layer1_model.predict(X_train_base)
X_train = np.column_stack([X_train_base, layer1_pred])

# Layer 2a
layer2a_model = MultiOutputRegressor(
    DecisionTreeRegressor(
        max_depth=dt_max_depth,
        min_samples_split=dt_min_samples_split,
        min_samples_leaf=dt_min_samples_leaf,
        random_state=42
    )
)
layer2a_model.fit(X_train, Y_train[:, [0, 1]])
layer2a_pred = layer2a_model.predict(X_train)

# Layer 2b
layer2b_model = MultiOutputRegressor(
    DecisionTreeRegressor(
        max_depth=dt_max_depth,
        min_samples_split=dt_min_samples_split,
        min_samples_leaf=dt_min_samples_leaf,
        random_state=42
    )
)
layer2b_model.fit(X_train, Y_train[:, [2, 3, 4]])
layer2b_pred = layer2b_model.predict(X_train)

X_train = np.column_stack([X_train, layer2a_pred, layer2b_pred])

# Layer 3
layer3_model = RandomForestRegressor(
    n_estimators=rf_n_estimators,
    max_depth=rf_max_depth,
    min_samples_split=rf_min_samples_split,
    min_samples_leaf=rf_min_samples_leaf,
    max_features=rf_max_features,
    random_state=42,
    n_jobs=-1
)
layer3_model.fit(X_train, Y_train)

# Evaluate on training data
Y_train_pred_norm = layer3_model.predict(X_train)
Y_train_denorm = denormalize_data(Y_train, output_norm_params, output_cols)
Y_train_pred_denorm = denormalize_data(Y_train_pred_norm, output_norm_params, output_cols)

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
    'layer1_model': layer1_model,
    'layer2a_model': layer2a_model,
    'layer2b_model': layer2b_model,
    'layer3_model': layer3_model,
    'output_norm_params': output_norm_params,
    'normalize_output_cols': output_cols,
    'feature_cols': feature_cols,
    'cv_results': cv_results
}

joblib.dump(model_data, model_path)


# ================================================== #
#    Test model
# ================================================== #
from Cascaded_ML_Test import evaluate_model
evaluate_model(model_path, test_data_path, plot_results=True)