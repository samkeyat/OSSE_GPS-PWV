import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

# Define the file pattern
file_pattern = "combined_Pseudo_Trans_SAM_hr_*.dat"

# Retrieve all files matching the pattern
files = sorted(glob.glob(file_pattern))

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Read and concatenate all files
for file in files:
    try:
        data = pd.read_csv(file, delim_whitespace=True, header=None,
                           names=["StationID", "Timestamp", "Superobbed", "NODA", 
                                  "50km", "100km", "200km", "NR", "REAL_OBS"])
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Convert the 'Timestamp' column to datetime
combined_data['Timestamp'] = pd.to_datetime(combined_data['Timestamp'])

# Handle missing data
combined_data = combined_data.dropna()

# Define stations with altitudes
stations = {
    "AZAM": 958.24, "AZBH": 181.07, "AZRV": 2132.79, "BIGE": 2572.44, "ECHO": 1706.33, 
    "ERAU": 1575.95, "GMPK": 624.46, "KEND": 1531.00, "KITT": 2090.19, "KP52": 1058.50, 
    "LUCK": 1370.00, "MING": 2291.30, "MORT": 1363.23, "MUND": 1949.35, "P001": 506.30, 
    "P003": 83.39, "P004": 1816.44, "P008": 1544.57, "P010": 1427.85, "P011": 1747.55, 
    "P014": 1098.61, "P015": 1953.67, "P026": 1260.31, "P107": 2011.30, "P623": 298.90, 
    "SA46": 761.58, "TNPP": 61.85
}

# Define altitude bands in a fixed order
bands = {
    "0-500": [], "500-1000": [], "1000-1500": [], "1500-2000": [], "2000+": []
}

# Categorize stations
for station, altitude in stations.items():
    if 0 <= altitude < 500:
        bands["0-500"].append(station)
    elif 500 <= altitude < 1000:
        bands["500-1000"].append(station)
    elif 1000 <= altitude < 1500:
        bands["1000-1500"].append(station)
    elif 1500 <= altitude < 2000:
        bands["1500-2000"].append(station)
    else:
        bands["2000+"].append(station)

# Print station assignments for each band
print("\n### Station Assignments by Altitude Band ###")
for band, station_list in bands.items():
    print(f"{band}: {', '.join(station_list) if station_list else 'None'}")

# Create a DataFrame to store bias statistics
bias_data = []

# Colors for different experiments
colors = {
    'Superobbed': 'r',
    '50km': 'k',
    '100km': 'b',
    '200km': 'purple',
    'NODA': 'orange'
}

# Create a figure with subplots in one column (number of rows = number of experiments)
num_experiments = 4  # You have 4 experiments to plot
fig, axes = plt.subplots(num_experiments, 1, figsize=(8, 12), sharex=True)
plt.rcParams.update({'font.size': 16})

# Flatten the axes array to make indexing easier (for cases where there are multiple rows)
axes = axes.flatten()

# Iterate through each experiment type and plot in a corresponding subplot
for i, experiment in enumerate(["Superobbed", "50km", "100km", "200km"]):
    bias_data = []  # Reset bias_data for each experiment

    for band, stations_list in bands.items():
        band_data = combined_data[combined_data['StationID'].isin(stations_list)]
        if band_data.empty:
            continue

        # Compute biases for the current experiment
        biases = band_data[experiment] - band_data['REAL_OBS']
        for value in biases:
            bias_data.append({'Altitude Band': band, 'Bias': value, 'Experiment': experiment})

    # Convert to DataFrame for the current experiment
    bias_df = pd.DataFrame(bias_data)

    # Ensure the correct order of altitude bands
    bias_df["Altitude Band"] = pd.Categorical(bias_df["Altitude Band"],
                                              categories=["0-500", "500-1000", "1000-1500", 
                                                          "1500-2000", "2000+"], ordered=True)

    # Sort DataFrame by altitude bands
    bias_df = bias_df.sort_values(by="Altitude Band")

    # Plot Box Plot with Line Plot for the current experiment
    ax = axes[i]  # Select the corresponding axis for this subplot

    # Create a boxplot
    bias_df.boxplot(column='Bias', by='Altitude Band', grid=False, showfliers=False, ax=ax)

    all_bands = ["0-500", "500-1000", "1000-1500", "1500-2000", "2000+"]
    # Compute median bias per altitude band
    median_values = bias_df.groupby('Altitude Band')['Bias'].median()

    # Ensure all bands exist in the median values (fill missing ones with NaN)
    median_values = median_values.reindex(all_bands, fill_value=np.nan)

    # Overlay line plot for the median bias trend inside the boxplot
    x_pos = [1, 2, 3, 4, 5]  # Positions for each altitude band
    ax.plot(x_pos, median_values, marker='o', linestyle='-', color='black', 
            label=f"Median Bias ({experiment})", linewidth=2, zorder=3)

    # Formatting for each subplot
    ax.set_title(f" {experiment}")
    ax.set_xlabel("Elevation Bin (m)")
    ax.set_ylabel("PWV Bias (mm)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis='both', labelsize=16)

    ax.set_ylim(-20,20)


# Adjust layout to make space between subplots
plt.tight_layout()

# Remove automatic title
plt.suptitle("")  

# Save and Show the figure with all subplots
plt.savefig('CASE01_all_experiment_bias_vs_altitude_ordered_final.jpg', dpi=360, bbox_inches='tight')
plt.show()
