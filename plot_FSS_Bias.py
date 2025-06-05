import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File list and labels
files = {
    "fss_bias_results_50km_multi_100km.csv": "50km",
    "fss_bias_results_100km_multi_100km.csv": "100km",
    "fss_bias_results_200km_multi_100km.csv": "200km",
    "fss_bias_results_superobbed_multi_100km.csv": "Superobbed",
    "fss_bias_results_NODA_multi_100km.csv": "NoDA",
    "fss_bias_results_NR_multi_100km.csv": "NR"
}

# Threshold columns (excluding 20 mm)
fss_columns = ["FSS_thr5_ws67", "FSS_thr10_ws67", "FSS_thr15_ws67"]
bias_columns = ["Bias_thr5_ws67", "Bias_thr10_ws67", "Bias_thr15_ws67"]
colors = ['black', 'blue', 'purple', 'red', 'orange', 'green']
methods = list(files.values())
n_methods = len(methods)

# Read files and gather timestamps
data = {}
original_timestamps = None
for file, label in files.items():
    df = pd.read_csv(file)
    if original_timestamps is None:
        original_timestamps = df['timestamp']
    data[label] = df.set_index('timestamp')

# Insert NaNs between every 4 timestamps to create wider spacing
group_size = 4
gap_size = 3  # Number of empty slots for spacing
timestamps_grouped = []
for i in range(0, len(original_timestamps), group_size):
    group = list(original_timestamps[i:i + group_size])
    timestamps_grouped.extend(group)
    if i + group_size < len(original_timestamps):
        timestamps_grouped.extend([''] * gap_size)

# Remove trailing empty labels
while timestamps_grouped and timestamps_grouped[-1] == '':
    timestamps_grouped.pop()

# Create x positions with gaps
x = np.arange(len(timestamps_grouped))
bar_width = 0.08
total_bars = 2 * n_methods
offset = (total_bars - 1) / 2 * bar_width

# Helper function to align the 0 point of ax2 with the 0 of ax
def align_zero(ax, ax2):
    y0_ax = ax.transData.transform((0, 0))[1]
    y0_ax2 = ax2.transData.transform((0, 0))[1]
    inv = ax2.transData.inverted()
    desired_0 = inv.transform((0, y0_ax))[1]
    current_ylim = ax2.get_ylim()
    offset_val = desired_0
    ax2.set_ylim(current_ylim[0] - offset_val, current_ylim[1] - offset_val)

# Set up figure and axes
fig, axs = plt.subplots(nrows=len(fss_columns), ncols=1, figsize=(20, 14), sharex=True)

for idx, (fss_col, bias_col) in enumerate(zip(fss_columns, bias_columns)):
    ax = axs[idx]
    ax2 = ax.twinx()

    for i, method in enumerate(methods):
        fss_values_all = data[method][fss_col].reindex(original_timestamps).fillna(0) * 100
        bias_values_all = (data[method][bias_col].reindex(original_timestamps).fillna(0) - 1) * 100

        fss_values_grouped = []
        bias_values_grouped = []
        for j in range(0, len(original_timestamps), group_size):
            fss_values_grouped.extend(fss_values_all.iloc[j:j + group_size])
            bias_values_grouped.extend(bias_values_all.iloc[j:j + group_size])
            if j + group_size < len(original_timestamps):
                fss_values_grouped.extend([np.nan] * gap_size)
                bias_values_grouped.extend([np.nan] * gap_size)

        fss_pos = x + (2 * i) * bar_width - offset
        bias_pos = x + (2 * i + 1) * bar_width - offset

        ax.bar(fss_pos, fss_values_grouped, width=bar_width,
               color=colors[i], alpha=0.8, label=method)

        ax2.bar(bias_pos, bias_values_grouped, width=bar_width,
                facecolor='none', edgecolor=colors[i], linewidth=1.5,
                hatch='//', label=f"{method} Bias")

    threshold = fss_col.split("thr")[1].split("_")[0]
    ax.set_title(f"Threshold: {threshold} mm", fontsize=22)
    ax.set_ylabel("FSS (%)", fontsize=22)
    ax2.set_ylabel("Bias (%)", fontsize=22)

    ax.set_ylim(-50, 50)
    ax2.set_ylim(-150, 150)
    align_zero(ax, ax2)

    # Set Bias y-ticks every 50%
    bias_ticks = np.arange(-150, 151, 50)
    ax2.set_yticks(bias_ticks)
    ax2.set_yticklabels([f'{int(val)}' for val in bias_ticks], fontsize=22)

    ax2.yaxis.set_tick_params(which='both', direction='in', length=6, width=2, colors='black')
    ax2.spines['right'].set_position(('outward', 0))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', labelsize=22)
    ax2.tick_params(axis='y', labelsize=22)

    if idx == 0:
        fss_legend = ax.legend(handles=ax.get_legend_handles_labels()[0],
                               labels=methods,
                               loc='upper center',
                               bbox_to_anchor=(0.5, 1.33),
                               fontsize=18,
                               ncol=len(methods),
                               frameon=False)

        bias_handles = [plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor=colors[i],
                                      linewidth=1.5, hatch='//') for i in range(len(methods))]
        bias_labels = [f"{method} Bias" for method in methods]
        bias_legend = ax.legend(handles=bias_handles,
                                labels=bias_labels,
                                loc='upper center',
                                bbox_to_anchor=(0.5, 1.23),
                                fontsize=18,
                                ncol=len(methods),
                                frameon=False)

        ax.add_artist(fss_legend)
        ax.add_artist(bias_legend)

# X-axis formatting
axs[-1].set_xticks(x)
axs[-1].set_xticklabels(timestamps_grouped, fontsize=20, ha='right')
axs[-1].set_xlabel("Timestamp", fontsize=22)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('FSS_Bias_100km.jpg', dpi=300)
