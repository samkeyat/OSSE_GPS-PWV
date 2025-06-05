import numpy as np
import os
import glob
import xarray as xr

# Constants
R_dryair = 287.0 
R_vapor = 461.0  
Rd_over_Rv = R_dryair / R_vapor
Cp_over_Cv = 1.4
g = 9.81

def calculate_pwv(ds):
    lon = ds.XLONG
    lat = ds.XLAT
    XLAT = ds['XLAT'][0,:,:]
    XLONG = ds['XLONG'][0,:,:]
    vert_eta_levels = ds['DNW'].squeeze()
    len_DNW = len(vert_eta_levels)
    QV_2m = ds['Q2']
    theta_t0 = ds['T']
    q_vapor = ds['QVAPOR'].squeeze()
    surface_pressure = ds['PSFC']
    perturb_geopotential = ds['PH'].squeeze()
    perturb_dry_air_mass_in_column = ds['MU']
    base_state_geopotential = ds['PHB'].squeeze()
    base_state_dry_air_mass_in_column = ds['MUB']

    qvf = 1.0 + q_vapor / Rd_over_Rv
    rho_pre = []

    for i in range(len_DNW):
        numerator = -(base_state_dry_air_mass_in_column + perturb_dry_air_mass_in_column)
        denominator = ((perturb_geopotential[i + 1, :, :] + base_state_geopotential[i + 1, :, :]) -
                       (perturb_geopotential[i, :, :] + base_state_geopotential[i, :, :])) / vert_eta_levels[i]
        rho_pre.append(numerator / denominator)

    rho = np.squeeze(np.asarray(rho_pre))
    tot_pres = 100000.0 * ((R_dryair * (300.0 + theta_t0) * qvf) / (100000.0 / rho))**Cp_over_Cv
    total_pressure = np.squeeze(np.asarray(tot_pres))
    PRESSURE_final = np.concatenate((surface_pressure, total_pressure), axis=0)
    QVAPOR_final = np.concatenate((QV_2m / (1.0 + QV_2m), q_vapor / (1.0 + q_vapor)), axis=0)   
    PWV_pre = []
    for z in range(len_DNW):
        z_term = []
        for j in range(perturb_dry_air_mass_in_column.shape[1]):
            row_term = []
            for i in range(perturb_dry_air_mass_in_column.shape[2]):
                term = 0.5 * abs (QVAPOR_final[z, j, i] + QVAPOR_final[z + 1, j, i]) * abs (PRESSURE_final[z, j, i] - PRESSURE_final[z + 1, j, i])
                row_term.append(term)
            z_term.append(row_term)
        PWV_pre.append(z_term)

    PWV = np.sum(np.asarray(PWV_pre), axis=0) / g
    return PWV, XLAT, XLONG
def process_wrf_file(wrf_file):
    ds = xr.open_dataset(wrf_file)
    PWV, XLAT, XLONG = calculate_pwv(ds)  # Retrieve XLAT and XLONG from calculate_pwv

    # Extract XTIME variable from the original dataset
    XTIME = ds['XTIME']

    # Extract dimensions from the original dataset
    dims = ('south_north', 'west_east')

    # Create an xarray Dataset with dimensions, PWV data, and XTIME variable
    ds_pwv = xr.Dataset({
        'PWV': (dims, PWV.data),
        'XLAT': (dims, XLAT.data),
        'XLONG': (dims, XLONG.data),
        'XTIME': ('Time', XTIME.data)
    })

    output_file_name = f"pwv_{os.path.basename(wrf_file)}"
    ds_pwv.to_netcdf(output_file_name)
    ds.close()
    ds_pwv.close()

def main():
    wrf_files = glob.glob('wrfout_d02*')

    for wrf_file in wrf_files:
        process_wrf_file(wrf_file)

if __name__ == "__main__":
    main()