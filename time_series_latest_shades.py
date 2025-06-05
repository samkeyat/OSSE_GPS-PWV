import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Define the file pattern
file_pattern = "combined_Pseudo_Trans_SAM_hr_*.dat"

# Retrieve all files matching the pattern
files = sorted(glob.glob(file_pattern))

# Initialize an empty DataFrame
combined_data = pd.DataFrame()

# Read and concatenate
for file in files:
    try:
        data = pd.read_csv(
            file, delim_whitespace=True, header=None,
            names=["StationID","Timestamp","OSSE_Superobbed","NODA",
                   "OSSE_50km","OSSE_100km","OSSE_200km","NR","REAL_OBS"]
        )
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Convert timestamp column
combined_data['Timestamp'] = pd.to_datetime(combined_data['Timestamp'])
combined_data = combined_data.dropna(
    subset=['Timestamp','OSSE_Superobbed','NODA','OSSE_50km',
            'OSSE_100km','OSSE_200km','NR','REAL_OBS']
)

# Safe correlation function
def safe_corrcoef(x, y):
    if np.std(x)==0 or np.std(y)==0:
        return np.nan
    return np.corrcoef(x,y)[0,1]

# Compute time‑series metrics (including NR)
time_series_metrics = combined_data.groupby('Timestamp').apply(lambda x: pd.Series({
    'Bias_OSSE_Superobbed': np.mean(x['OSSE_Superobbed'] - x['REAL_OBS']),
    'Bias_OSSE_50km':     np.mean(x['OSSE_50km']     - x['REAL_OBS']),
    'Bias_OSSE_100km':    np.mean(x['OSSE_100km']    - x['REAL_OBS']),
    'Bias_OSSE_200km':    np.mean(x['OSSE_200km']    - x['REAL_OBS']),
    'Bias_NODA':          np.mean(x['NODA']          - x['REAL_OBS']),
    'Bias_NR':            np.mean(x['NR']             - x['REAL_OBS']),

    'RMSE_OSSE_Superobbed': np.sqrt(mean_squared_error(x['REAL_OBS'], x['OSSE_Superobbed'])),
    'RMSE_OSSE_50km':       np.sqrt(mean_squared_error(x['REAL_OBS'], x['OSSE_50km'])),
    'RMSE_OSSE_100km':      np.sqrt(mean_squared_error(x['REAL_OBS'], x['OSSE_100km'])),
    'RMSE_OSSE_200km':      np.sqrt(mean_squared_error(x['REAL_OBS'], x['OSSE_200km'])),
    'RMSE_NODA':            np.sqrt(mean_squared_error(x['REAL_OBS'], x['NODA'])),
    'RMSE_NR':              np.sqrt(mean_squared_error(x['REAL_OBS'], x['NR'])),

    'Correlation_OSSE_Superobbed': safe_corrcoef(x['REAL_OBS'], x['OSSE_Superobbed']),
    'Correlation_OSSE_50km':       safe_corrcoef(x['REAL_OBS'], x['OSSE_50km']),
    'Correlation_OSSE_100km':      safe_corrcoef(x['REAL_OBS'], x['OSSE_100km']),
    'Correlation_OSSE_200km':      safe_corrcoef(x['REAL_OBS'], x['OSSE_200km']),
    'Correlation_NODA':            safe_corrcoef(x['REAL_OBS'], x['NODA']),
    'Correlation_NR':              safe_corrcoef(x['REAL_OBS'], x['NR']),
}))

# Compute mean PWV for plotting
mean_values = combined_data.groupby('Timestamp')[[
    'REAL_OBS','OSSE_Superobbed','OSSE_50km',
    'OSSE_100km','OSSE_200km','NODA','NR'
]].mean()

# Colors
colors = {
    'REAL_OBS': 'm',
    'OSSE_Superobbed': 'r',
    'OSSE_50km': 'k',
    'OSSE_100km': 'b',
    'OSSE_200km': 'purple',
    'NODA': 'orange',
    'NR': 'g'
}

# Shade window
shade_start = datetime(2021,8,15,12)
shade_end   = datetime(2021,8,15,18)

# Create figure
plt.figure(figsize=(16,10))
plt.rcParams.update({'font.size':18})

# 1) Mean
plt.subplot(2,2,1)
plt.axvspan(shade_start,shade_end,color='gray',alpha=0.3)
for col in ['REAL_OBS','OSSE_Superobbed','OSSE_50km','OSSE_100km','OSSE_200km','NODA','NR']:
    ls = '--' if col=='REAL_OBS' else '-'
    lw = 3 if col=='REAL_OBS' else 1
    plt.plot(mean_values.index, mean_values[col],
             label=col, color=colors[col], linestyle=ls, linewidth=lw)
plt.title('Mean')
plt.ylabel('PWV (mm)')
plt.xlabel('Timestamp')
plt.legend(fontsize=12,loc='upper right')
plt.grid(True,linestyle='--',alpha=0.5)
plt.xticks(rotation=45)
plt.ylim(25,40)

# 2) Bias
plt.subplot(2,2,2)
plt.axvspan(shade_start,shade_end,color='gray',alpha=0.3)
for col in ['Bias_OSSE_Superobbed','Bias_OSSE_50km','Bias_OSSE_100km','Bias_OSSE_200km','Bias_NODA','Bias_NR']:
    sim = col.replace('Bias_','')
    plt.plot(time_series_metrics.index, time_series_metrics[col],
             label=sim, color=colors[sim])
plt.title('Bias')
plt.ylabel('Bias (mm)')
plt.xlabel('Timestamp')
plt.grid(True,linestyle='--',alpha=0.5)
plt.xticks(rotation=45)
plt.ylim(-6,3)

# 3) RMSE
plt.subplot(2,2,3)
plt.axvspan(shade_start,shade_end,color='gray',alpha=0.3)
for col in ['RMSE_OSSE_Superobbed','RMSE_OSSE_50km','RMSE_OSSE_100km','RMSE_OSSE_200km','RMSE_NODA','RMSE_NR']:
    sim = col.replace('RMSE_','')
    plt.plot(time_series_metrics.index, time_series_metrics[col],
             label=sim, color=colors[sim])
plt.title('RMSE')
plt.ylabel('RMSE (mm)')
plt.xlabel('Timestamp')
plt.grid(True,linestyle='--',alpha=0.5)
plt.xticks(rotation=45)
plt.ylim(2,10)

# 4) Correlation
plt.subplot(2,2,4)
plt.axvspan(shade_start,shade_end,color='gray',alpha=0.3)
for col in ['Correlation_OSSE_Superobbed','Correlation_OSSE_50km','Correlation_OSSE_100km',
            'Correlation_OSSE_200km','Correlation_NODA','Correlation_NR']:
    sim = col.replace('Correlation_','')
    plt.plot(time_series_metrics.index, time_series_metrics[col],
             label=sim, color=colors[sim])
plt.title('Correlation')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Timestamp')
plt.grid(True,linestyle='--',alpha=0.5)
plt.xticks(rotation=45)
plt.ylim(0.7,1)

plt.tight_layout()
# Save
plt.savefig('time_series_metrics_with_pwv_and_NR.jpg',dpi=360,bbox_inches='tight')
plt.savefig('time_series_metrics_with_pwv_and_NR.pdf',bbox_inches='tight')
plt.show()
