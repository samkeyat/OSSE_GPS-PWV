import numpy as np
import glob
from scipy.interpolate import griddata
import xarray as xr


# Define the bilinear interpolation function
def bilinear_interpolation(x, y, lon, lat, data):
    points = np.column_stack((lon.values.ravel(), lat.values.ravel()))
    values = data.values.ravel()
    interp_values = griddata(points, values, (x, y), method='linear')
    return interp_values

# Read station information from the text file
station_data = []

with open("GPS_DAVID-SUOMI_sites_2021.txt", "r") as station_file:
    for line in station_file:
        parts = line.split()
        if len(parts) >= 3:  # Ensure there's at least one word after splitting
            lon, lat, *name_parts = parts
            lon = float(lon)
            lat = float(lat)
            name = ' '.join(name_parts)  # Join all parts into the station name
            station_data.append((lon, lat, name))

# Define the output file
output_file = 'Synthetic_GPS_MET_station_pwv_CASE01_mem02_syn_st_50km.dat'

# Loop through two days (from 2021-08-15 to 2021-08-16)
for day in range(2):
    for hour in range(24):
        date = f'2021-08-15' if day == 0 else f'2021-08-16'
        file_pattern = f'pwv_wrfout_d02_{date}_{hour:02d}%3A00%3A00'
        matching_files = glob.glob(file_pattern)

        if not matching_files:
            continue  # Skip if no matching files are found

        # Load the first matching file to get the time
        data = xr.open_dataset(matching_files[0])
        time = data.XTIME.values

        with open(output_file, 'a') as outfile:
            for lon, lat, name in station_data:
                pwv = bilinear_interpolation(lon, lat, data.XLONG, data.XLAT, data.PWV)
                pwv_value = pwv  # Assign the interpolated value directly
                outfile.write(f"{name},{lon},{lat},{time},{pwv_value}\n")