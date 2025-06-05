import netCDF4 as nc
import glob

# Get a list of all NetCDF files in the directory
file_list = glob.glob('wrfout_d02*')

# Sort the file list to ensure proper order
file_list.sort()

# Iterate over each pair of consecutive files
for i in range(len(file_list) - 1):
    # Open the two input files
    file1 = nc.Dataset(file_list[i])
    file2 = nc.Dataset(file_list[i+1])

    # Get the variable objects
    var1 = file1.variables['RAINNC']
    var2 = file2.variables['RAINNC']

    # Subtract one variable from the other
    result = var2[:] - var1[:]

    # Create a new NetCDF file
    output_file =  'NR_hourly_' + file_list[i+1].split('/')[-1]+'.nc'

    with nc.Dataset(output_file, 'w', format='NETCDF4') as out_file:
        # Copy dimensions from one of the input files
        for dimname, dim in file1.dimensions.items():
            out_file.createDimension(dimname, len(dim) if not dim.isunlimited() else None)

        # Create variables in the output file
        out_var = out_file.createVariable('RAINNC', var1.datatype, var1.dimensions)

        # Copy attributes from one of the input variables
        out_var.setncatts({k: var1.getncattr(k) for k in var1.ncattrs()})

        # Write the result to the output file
        out_var[:] = result[:]

        # Copy XLAT and XLONG variables
        for var_name in ['XLAT', 'XLONG']:
            if var_name in file1.variables:
                var = file1.variables[var_name]
                out_var = out_file.createVariable(var_name, var.datatype, var.dimensions)
                out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                out_var[:] = var[:]

    # Close the files
    file1.close()
    file2.close()
