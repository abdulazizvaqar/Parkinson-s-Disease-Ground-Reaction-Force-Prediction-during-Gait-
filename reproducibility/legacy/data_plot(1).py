import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import glob
from matplotlib.gridspec import GridSpec

# Path to the Excel file
file_path = 'Output.xlsx'

# Sheet names to import
# sheet_names = ["AP-L", "ML-L", "Long-L"]

# sheet_name = 'AP-L'
sheet_name = 'ML-L'
# sheet_name = 'Long-L'

# Create a figure with a grid spec for plotting and tables
sns.set(style='whitegrid')  # Set aesthetic style
fig = plt.figure(constrained_layout=True, figsize=(14, 10))
gs = GridSpec(2, 1, height_ratios=[2, 1], figure=fig)

# Create subplots
ax = fig.add_subplot(gs[0])
# table_ax = fig.add_subplot(gs[1])

# Initialize dictionary to store data for statistical analysis
sheet_data = {}

# Iterate over each sheet and plot the data
# for sheet_name in sheet_names:
# Read the Excel sheet
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Extract median, 25th percentile, and 75th percentile from columns
median = df.iloc[:, 0]
percentile_25 = df.iloc[:, 1]
percentile_75 = df.iloc[:, 2]

# Plot the data with fill between percentiles
ax.plot(median, label=sheet_name)
ax.fill_between(range(len(median)), percentile_25, percentile_75, alpha=0.3)

# Store data for statistical analysis
sheet_data[sheet_name] = median

# Set chart title and labels
ax.set_title('Medio-Lateral', fontsize=26, fontweight='bold')
ax.set_ylabel('Force (N/kg)', fontsize=24)
ax.set_xlabel('Gait Cycle (%)', fontsize=24)
ax.legend(fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=24)



# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('ML-Force.png', dpi=600)
plt.show()