import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import glob
from matplotlib.gridspec import GridSpec
import numpy as np

# Path to the Excel file


#file_path = 'APLEFT.xlsx'
#file_path = 'APRIGHT.xlsx'
# file_path = 'MLLEFT.xlsx'
#file_path = 'MLRIGHT.xlsx'
#file_path = 'LONGLEFT.xlsx'
file_path = 'LONGRIGHT.xlsx'

excel_file = pd.ExcelFile(file_path)

# List all sheet names
sheet_names = excel_file.sheet_names


# Sheet names to import
# sheet_names = ["Measured", "1D-CONVO", "GRU", "LSTM"]

# Create a figure with a grid spec for plotting and tables
sns.set(style='whitegrid')  # Set aesthetic style
fig = plt.figure(constrained_layout=True, figsize=(14, 10))
gs = GridSpec(2, 1, height_ratios=[2, 1], figure=fig)

# Create subplots
ax = fig.add_subplot(gs[0])
table_ax = fig.add_subplot(gs[1])

# Initialize dictionary to store data for statistical analysis
sheet_data = {}

# Iterate over each sheet and plot the data
for sheet_name in ['MEASURED',
 'ANKLE',
 'HEEL',
 'KNEE',
 'ANK-HEEL',
 'ANK-KNEE',
 'ANKLE-HIP']:
    # Read the Excel sheet
    # sheet_name = "Measured"
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Interpolate each column to have exactly 100 rows
    num_rows = 100
    x_old = np.linspace(0, len(df) - 1, len(df))  # Original index range
    x_new = np.linspace(0, len(df) - 1, num_rows)  # New index range
    
    interpolated_df = pd.DataFrame(
        {col: np.interp(x_new, x_old, df[col].values) for col in df.columns}
    )
    
    # Compute median and percentiles across all rows
    median = interpolated_df.median(axis=1)  # Median across all rows
    percentile_25 = interpolated_df.quantile(0.25, axis=1)  # 25th percentile
    percentile_75 = interpolated_df.quantile(0.75, axis=1)  # 75th percentile
    
    # Plot the data with fill between percentiles
    ax.plot(median, label=sheet_name)
    ax.fill_between(range(num_rows), percentile_25, percentile_75, alpha=0.3)
    
    # Store data for statistical analysis
    sheet_data[sheet_name] = median

# Set chart title and labels
ax.set_title('LONG RIGHT GRF', fontsize=20, fontweight='bold')
ax.set_ylabel('Force (kg/BW)', fontsize=18)
ax.set_xlabel('Gait Cycle (%)', fontsize=18)
ax.legend(fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=18)

# Perform pairwise statistical analysis and correlation analysis between Measured and other sheets
statistical_results = []
base_data = sheet_data['MEASURED']

for sheet_name in ['ANKLE',
 'HEEL',
 'KNEE',
 'ANK-HEEL',
 'ANK-KNEE',
 'ANKLE-HIP']:
    comparison_data = sheet_data[sheet_name]
    
    # Perform statistical analysis (Mann-Whitney U test)
    stat, p_value = stats.mannwhitneyu(base_data, comparison_data)
    significance = '**' if p_value < 0.05 else ''
    
    # Perform correlation analysis (Spearman correlation)
    correlation, p_value_corr = stats.spearmanr(base_data, comparison_data)
    significance_corr = '**' if p_value_corr < 0.05 else ''
    
    # Append results to statistical_results
    statistical_results.append({
        'Comparison': f'Measured vs {sheet_name}',
        'p-value': f'{p_value:.4f}{significance}',
        'Correlation Coefficient': f'{correlation:.4f}',
        # 'Correlation p-value': f'{p_value_corr:.4f}{significance_corr}'
    })

# Convert statistical results to DataFrame
results_df = pd.DataFrame(statistical_results)

# Display the table on the plot
table_ax.axis('off')  # Hide the axis
cell_text = results_df.values
table = table_ax.table(cellText=cell_text, colLabels=results_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(1, 1.5)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(f'{os.path.basename(file_path)[:-5]}.png', dpi=1200)
plt.show()