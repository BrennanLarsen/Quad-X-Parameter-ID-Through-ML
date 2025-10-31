# DecisionTree_Test.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')


# ================================================== #
#    Paths
# ================================================== #
model_path = Path(__file__).parent / "DecisionTree_Model.pkl"
test_data_path = Path(
    r"TEST_DATA_PATH_HERE.xlsx"
)


# ================================================== #
#    Helper Functions
# ================================================== #
def normalize_data(data, norm_params, columns):  # min-max normalization using stored normalization params
    data_normalized = data.copy()
    for col in columns:
        if col in norm_params:
            min_val = norm_params[col]['min']
            range_val = norm_params[col]['range']
            if range_val == 1 and min_val == norm_params[col]['max']:
                data_normalized[col] = 0
            else:
                data_normalized[col] = (data[col] - min_val) / range_val
    return data_normalized


def denormalize_data(data_normalized, norm_params, columns):  # reverse normalization using stored normalization params
    data_denormalized = data_normalized.copy()
    for i, col in enumerate(columns):
        min_val = norm_params[col]['min']
        range_val = norm_params[col]['range']
        data_denormalized[:, i] = data_normalized[:, i] * range_val + min_val
    return data_denormalized


def create_quadrotor_features(sequence):  # Feature engineering
    mean_vals = np.mean(sequence, axis=0)
    std_vals = np.std(sequence, axis=0)
    min_vals = np.min(sequence, axis=0)
    max_vals = np.max(sequence, axis=0)
    first_vals = sequence[0, :]
    last_vals = sequence[-1, :]
    range_vals = max_vals - min_vals
    median_vals = np.median(sequence, axis=0)
    features = np.concatenate([
        mean_vals, std_vals, min_vals, max_vals,
        first_vals, last_vals, range_vals, median_vals
    ])
    return features


# ================================================== #
#    Load and Prep Test Data
# ================================================== #
def load_test_data(path, sequence_length, input_norm_params, output_norm_params,
                   normalize_input_cols, normalize_output_cols):
    df_test = pd.read_excel(path, engine='openpyxl')
    input_cols = ['t', 'x', 'y', 'z', 'dx', 'dy', 'dz', 'ddx', 'ddy', 'ddz', 'phi', 'theta', 'psi',
                  'p', 'q', 'r', 'dp', 'dq', 'dr', 'omega_1', 'omega_2', 'omega_3', 'omega_4', 'm']
    output_cols = ['c_T', 'c_RD', 'I_x', 'I_y', 'I_z']

    df_inputs = df_test[input_cols].copy()
    df_outputs = df_test[output_cols].copy()

    df_inputs_norm = normalize_data(df_inputs, input_norm_params, normalize_input_cols)
    df_outputs_norm = normalize_data(df_outputs, output_norm_params, normalize_output_cols)

    num_quadrotors = len(df_test) // sequence_length
    X_test, Y_test = [], []

    for i in range(num_quadrotors):
        start_idx = i * sequence_length
        end_idx = start_idx + sequence_length
        flight_inputs = df_inputs_norm.values[start_idx:end_idx, :]
        parameters = df_outputs_norm.values[start_idx, :]
        features = create_quadrotor_features(flight_inputs)
        X_test.append(features)
        Y_test.append(parameters)

    return np.array(X_test), np.array(Y_test), num_quadrotors


# ================================================== #
#    Evaluation Function
# ================================================== #
def evaluate_model(model_path, test_data_path, plot_results=True):
    """Evaluate trained model performance on unseen test data."""
    model_data = joblib.load(model_path)
    model = model_data['model']
    hp = model_data['hyperparameters']
    input_norm_params = model_data['input_norm_params']
    output_norm_params = model_data['output_norm_params']
    normalize_input_cols = model_data['normalize_input_cols']
    normalize_output_cols = model_data['normalize_output_cols']

    sequence_length = hp.get("sequence_length", 297)

    X_test, Y_test_norm, num_quadrotors = load_test_data(
        test_data_path, sequence_length, input_norm_params, output_norm_params,
        normalize_input_cols, normalize_output_cols
    )

    predictions_norm = model.predict(X_test)
    Y_test = denormalize_data(Y_test_norm, output_norm_params, normalize_output_cols)
    predictions = denormalize_data(predictions_norm, output_norm_params, normalize_output_cols)

    output_names = ["Thrust Coeff", "Drag Coeff", "X-Inertia", "Y-Inertia", "Z-Inertia"]

    print("\nTEST SET MEAN PERCENT ERROR:")
    print("=" * 50)
    for i, name in enumerate(output_names):
        errors = predictions[:, i] - Y_test[:, i]
        percent_errors = (errors / Y_test[:, i]) * 100
        mean_percent_error = np.mean(np.abs(percent_errors))
        print(f"{name:<25} {mean_percent_error:>6.2f}%")

    if plot_results:
        fig, axes = plt.subplots(5, 2, figsize=(14, 16))
        axes[0, 0].set_title('Predicted Values', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Percent Error', fontsize=12, fontweight='bold')

        for i, name in enumerate(output_names):
            quad_indices = np.arange(num_quadrotors)

            ax_pred = axes[i, 0]
            ax_pred.plot(quad_indices, Y_test[:, i], 'bo-', linewidth=1.2, markersize=3, label='Actual')
            ax_pred.plot(quad_indices, predictions[:, i], 'rs--', linewidth=1.2, markersize=3, label='Predicted')
            if i == len(output_names) - 1:
                ax_pred.set_xlabel('Quadrotor Index', fontsize=8)
            ax_pred.set_ylabel(f'{name}', fontsize=8)
            ax_pred.legend(loc='best')
            ax_pred.grid(True)

            ax_err = axes[i, 1]
            errors = predictions[:, i] - Y_test[:, i]
            percent_errors = (errors / Y_test[:, i]) * 100
            mean_percent_error = np.mean(np.abs(percent_errors))
            ax_err.plot(quad_indices, percent_errors, 'go-', linewidth=1.2, markersize=3)
            ax_err.axhline(y=0, color='k', linestyle='-', linewidth=1)
            ax_err.axhline(y=mean_percent_error, color='k', linestyle='--', linewidth=1.2, 
                           label=f'Avg: {mean_percent_error:.2f}%')
            if i == len(output_names) - 1:
                ax_err.set_xlabel('Quadrotor Index', fontsize=10)
            ax_err.set_ylabel(f'{name}', fontsize=10)
            ax_err.legend(loc='upper right')
            ax_err.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    evaluate_model(model_path, test_data_path, plot_results=True)
