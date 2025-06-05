import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import glob
import re

# Define file patterns

wrf_files = sorted(glob.glob("100km_hourly*.nc"))  # Adjust pattern if needed
mrms_file = "MRMS_accumulated_precip_1518_1606.nc"

# Open MRMS dataset (fixed grid for all iterations)
mrms_ds = xr.open_dataset(mrms_file)
mrms_lons, mrms_lats = np.meshgrid(mrms_ds["lon_0"], mrms_ds["lat_0"])

# Loop over multiple WRF files
for wrf_file in wrf_files:
    print(f"Processing {wrf_file}...")

    # Open WRF dataset
    wrf_ds = xr.open_dataset(wrf_file)

    # Extract WRF precipitation data and coordinates
    wrf_data = wrf_ds["RAINNC"].squeeze()  # Remove time dimension if present
    wrf_lons, wrf_lats = wrf_ds["XLONG"].squeeze(), wrf_ds["XLAT"].squeeze()

    # Convert WRF longitudes from [-180, 180] to [0, 360] format
    wrf_lons = wrf_lons % 360

    # Flatten the WRF grid and data for interpolation
    wrf_points = np.column_stack((wrf_lons.values.ravel(), wrf_lats.values.ravel()))
    wrf_values = wrf_data.values.ravel().astype(float)

    # Remove NaN values before interpolation
    valid_mask = ~np.isnan(wrf_values)
    wrf_points = wrf_points[valid_mask]
    wrf_values = wrf_values[valid_mask]

    # Interpolate using nearest-neighbor method
    mrms_points = np.column_stack((mrms_lons.ravel(), mrms_lats.ravel()))
    regridded_values = griddata(wrf_points, wrf_values, mrms_points, method='nearest')

    # Reshape back to MRMS grid dimensions
    regridded_wrf = xr.DataArray(
        regridded_values.reshape(mrms_lats.shape),
        dims=("lat_0", "lon_0"),
        coords={"lat_0": mrms_ds["lat_0"], "lon_0": mrms_ds["lon_0"]},
        name="RAINNC"
    )

    # Copy attributes from the original data
    regridded_wrf.attrs = wrf_ds["RAINNC"].attrs

    # Update the regular expression to account for the URL-encoded colon (%3A)
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})%3A(\d{2})%3A(\d{2})", wrf_file)
    if match:
        date_str = match.group(1)
        hour_str = match.group(2)
        minute_str = match.group(3)
        second_str = match.group(4)
        
        # Format the output filename to include date and time
        date_hour_str = f"{date_str}_{hour_str}{minute_str}{second_str}"
        output_filename = f"100km_wrf_regridded_to_mrms_{date_hour_str}.nc"
    else:
        # Fallback to default naming if the pattern is not found
        output_filename = f"wrf_regridded_to_mrms_{wrf_file.split('_')[-1].replace('.nc', '.nc')}"

    # Save output with a unique filename
    regridded_ds = xr.Dataset({"RAINNC": regridded_wrf})
    regridded_ds.to_netcdf(output_filename)

    print(f"Saved: {output_filename}")

print("Processing complete!")
