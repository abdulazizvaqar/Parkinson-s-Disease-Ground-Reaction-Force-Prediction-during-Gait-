import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import glob
import matplotlib.patches as mpatches

# Define a function to perform pairwise statistical analysis between columns
# Define a function to perform pairwise statistical analysis between columns
def analyze_columns(data, ax):
    columns = data.columns
    normality = [stats.shapiro(data[col])[1] > 0.05 for col in columns]
    p_values = []
    line_offsets = {}  # Track offsets for each pair to avoid overlap

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            if normality[i] and normality[j]:
                # If both columns are normally distributed, apply t-test
                stat, p_value = stats.ttest_ind(data[col1], data[col2])
            else:
                # If not both columns are normally distributed, apply Mann-Whitney U test
                stat, p_value = stats.mannwhitneyu(data[col1], data[col2])

            # Store p-value for the table
            p_values.append((col1, col2, p_value))

            # Add significance level markers and bracket to the plot if there is a significant difference
            if p_value < 0.05:
                y_max = max(data[col1].max(), data[col2].max())
                significance_marker = 'n.s.'
                if p_value < 0.001:
                    significance_marker = '***'
                elif p_value < 0.01:
                    significance_marker = '**'
                elif p_value < 0.05:
                    significance_marker = '*'

                # Determine the line offset for this pair
                offset_key = (i, j)
                if offset_key not in line_offsets:
                    line_offsets[offset_key] = len(line_offsets) * 0.1  # Increment offset

                y_line = y_max * (1.18 + line_offsets[offset_key])

                ax.text((i + j) / 2, y_line + 0.05, significance_marker, ha='center', va='bottom', fontsize=18, color='red')
                ax.plot([i, j], [y_line, y_line], color='black', linewidth=1.5)

    return p_values

# Path to the folder containing Excel files
folder_path = ''
excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))

# Create a figure for plotting
sns.set(style='whitegrid')  # Set aesthetic style

# Iterate over each Excel file and plot individual boxplots
for file in excel_files:
    # Read the Excel file
    xls = pd.ExcelFile(file)
    
    if 'NRMSE' in xls.sheet_names:
        df = pd.read_excel(file, sheet_name='NRMSE')
        
        # Create a new figure for each plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot boxplot for the three columns
        sns.boxplot(data=df, ax=ax, palette='Set2')
        ax.set_title('NRMSE Vertical GRF', fontsize=20, fontweight='bold')
        ax.set_ylabel('NRMSE', fontsize=18)
        ax.set_xlabel('Input', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)

        # Set y-axis limit
        ax.set_ylim(0, 1)

        # Add individual data points
        for i, col in enumerate(df.columns):
            y = df[col]
            x = [i] * len(y)
            ax.scatter(x, y, alpha=0.6, edgecolors='w', linewidth=0.5)

        # Perform statistical analysis and add results to the plot
        analyze_columns(df, ax)

        # Save individual plot
        plt.tight_layout()
        plt.savefig(f'NRMSE_boxplot_analysis_{os.path.basename(file)[:-5]}.png', dpi=1200)
        plt.show()