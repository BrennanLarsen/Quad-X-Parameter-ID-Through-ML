import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ================================================== #
#    Data
# ================================================== #
data_path = Path(__file__).parent / "DATA_PATH_HERE.xlsx"
df = pd.read_excel(data_path)

# ================================================== #
#    Columns to check correlations
# ================================================== #
cols_to_check = ['t','x','y','z','dx','dy','dz','ddx','ddy','ddz','phi','theta','psi','p','q','r','dp','dq','dr',
              'omega_1','omega_2','omega_3','omega_4','m', 'c_T','c_RD','I_x','I_y','I_z']

# ================================================== #
#    Pearson correlation
# ================================================== #
corr_matrix = df[cols_to_check].corr(method="pearson")


# ================================================== #
#    Plot heatmap
# ================================================== #
plt.xticks(rotation=45, ha='right', fontsize=8)   
plt.yticks(rotation=0, fontsize=8)                
sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap="coolwarm", cbar=True, square=False, annot_kws={"size": 6})
plt.title("Pearson Correlation Map")
plt.tight_layout()
plt.show()

