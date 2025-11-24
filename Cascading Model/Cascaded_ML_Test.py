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
model_path = Path(__file__).parent / "Cascading_Model.pkl"
test_data_path = Path(
    r"TEST_DATA_FILE_PATH_HERE.xlsx"
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


# ================================================== #
#    Load and Prep Test Data
# ================================================== #
def load_test_data(path, feature_cols, output_norm_params, normalize_output_cols):
    df_test = pd.read_excel(path, engine='openpyxl')
    output_cols = ['c_T', 'c_RD', 'I_x', 'I_y', 'I_z']

    # Extract features
    X_test = df_test[feature_cols].values

    # Extract and normalize outputs
    df_outputs = df_test[output_cols].copy()
    df_outputs_norm = normalize_data(df_outputs, output_norm_params, normalize_output_cols)

    num_samples = len(df_test)

    return X_test, df_outputs_norm.values, num_samples


# ================================================== #
#    Evaluation Function
# ================================================== #
def evaluate_model(model_path, test_data_path, plot_results=True):
    model_data = joblib.load(model_path)
    layer1_model = model_data['layer1_model']
    layer2a_model = model_data['layer2a_model']
    layer2b_model = model_data['layer2b_model']
    layer3_model = model_data['layer3_model']
    output_norm_params = model_data['output_norm_params']
    normalize_output_cols = model_data['normalize_output_cols']
    feature_cols = model_data['feature_cols']

    X_test_base, Y_test_norm, num_samples = load_test_data(
        test_data_path, feature_cols, output_norm_params, normalize_output_cols
    )

    # Layer 1
    layer1_pred = layer1_model.predict(X_test_base)
    X_test_L2 = np.column_stack([X_test_base, layer1_pred])
    
    # Layer 2a
    layer2a_pred = layer2a_model.predict(X_test_L2)
    
    # Layer 2b
    layer2b_pred = layer2b_model.predict(X_test_L2)
    
    # Layer 3
    X_test_L3 = np.column_stack([X_test_L2, layer2a_pred, layer2b_pred])
    predictions_norm = layer3_model.predict(X_test_L3)
    
    # denormalize predictions and actuals
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
            sample_indices = np.arange(num_samples)

            ax_pred = axes[i, 0]
            ax_pred.plot(sample_indices, Y_test[:, i], 'bo-', linewidth=1.2, 
                        markersize=3, label='Actual')
            ax_pred.plot(sample_indices, predictions[:, i], 'rs--', linewidth=1.2, 
                        markersize=3, label='Predicted')
            if i == 4:
                ax_pred.set_xlabel('Sample Index', fontsize=10)
            ax_pred.set_ylabel(f'{name}', fontsize=10)
            ax_pred.legend(loc='upper left')
            ax_pred.grid(True)

            ax_err = axes[i, 1]
            errors = predictions[:, i] - Y_test[:, i]
            percent_errors = (errors / Y_test[:, i]) * 100
            mean_percent_error = np.mean(np.abs(percent_errors))
            ax_err.plot(sample_indices, percent_errors, 'go-', linewidth=1.2, markersize=3)
            ax_err.axhline(y=0, color='k', linestyle='-', linewidth=1)
            ax_err.axhline(y=mean_percent_error, color='k', linestyle='--', linewidth=1.2, 
                          label=f'Avg: {mean_percent_error:.2f}%')
            if i == 4:
                ax_err.set_xlabel('Sample Index', fontsize=10)
            ax_err.set_ylabel(f'{name}', fontsize=10)
            ax_err.legend(loc='upper right')
            ax_err.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    evaluate_model(model_path, test_data_path, plot_results=True)