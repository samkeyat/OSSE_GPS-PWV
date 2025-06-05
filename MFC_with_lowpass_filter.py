import warnings
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
import cartopy.crs as ccrs
import cartopy.feature as cf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset
import matplotlib.colors
from wrf import getvar, latlon_coords, interplevel, to_np
from scipy.ndimage import gaussian_filter

# Suppress warnings
warnings.filterwarnings('ignore')

# Define the file pattern to search for multiple NetCDF files
file_pattern = 'wrfout_d02_*'
nc_files = glob.glob(file_pattern)

if not nc_files:
    print("No NetCDF files matching the pattern were found.")
else:
    for file in nc_files:
        # Extract time from filename (modify based on your filename structure)
        filename = file.split('/')[-1]
        time_str = filename.split('_')[2] + "_" + filename.split('_')[3].split('%')[0].replace('%3A', ':')

        with Dataset(file, 'r') as nc:
            # Extract fields
            qvapor = getvar(nc, "QVAPOR")
            ua = getvar(nc, "ua")
            va = getvar(nc, "va")
            pressures = getvar(nc, "P") + getvar(nc, "PB")
            target_pressure = 70000.0

            # Interpolate to 700 mb
            qv_700 = interplevel(qvapor, pressures, target_pressure)
            ua_700 = interplevel(ua, pressures, target_pressure)
            va_700 = interplevel(va, pressures, target_pressure)

            # Get lat/lon
            lat, lon = latlon_coords(qv_700)

            # Compute moisture flux and convergence
            q_flux_u = qv_700 * ua_700
            q_flux_v = qv_700 * va_700
            dqdx = np.gradient(to_np(q_flux_u), axis=1)
            dqdy = np.gradient(to_np(q_flux_v), axis=0)
            flux_convergence = -(dqdx + dqdy)

            # Apply Gaussian low-pass filter
            flux_convergence_smooth = gaussian_filter(flux_convergence, sigma=2)

            # Near-surface wind for plotting
            u10 = getvar(nc, "U10")
            v10 = getvar(nc, "V10")
            lats_ns, lons_ns = latlon_coords(u10)
            u10_np = to_np(u10)
            v10_np = to_np(v10)
            lats_np = to_np(lats_ns)
            lons_np = to_np(lons_ns)

            # Plot
            fig = plt.figure(figsize=(12, 8))
            plt.tight_layout()
            ax = plt.subplot(111, projection=ccrs.PlateCarree())
            mplt.rc('xtick', labelsize=12)
            mplt.rc('ytick', labelsize=12)
            plt.rcParams.update({'font.size': 22})

            ax.add_feature(cf.BORDERS, linewidth=0.5, edgecolor="k")
            ax.coastlines('50m', linewidth=0.8)
            ax.add_feature(cf.STATES, edgecolor="k")
            ax.set_extent([-115, -107, 37.2, 31.1])
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                              linewidth=1, color='black', alpha=0.5, linestyle='--')

            # Cities
            cities = {
                'Phoenix': (-112.07, 33.44),
                'Tucson': (-110.97, 32.25),
                'Prescott': (-112.46, 34.54),
                'Flagstaff': (-111.65, 35.19)
            }
            for city, coordinates in cities.items():
                ax.plot(coordinates[0], coordinates[1], 'kD', transform=ccrs.PlateCarree())
                ax.text(coordinates[0] - 0.2, coordinates[1] - 0.2, city,
                        transform=ccrs.PlateCarree(), fontsize=18)

            gl.bottom_labels = False
            gl.left_labels = False

            # Colormap and levels
            cmap = plt.cm.RdBu_r
            levels = np.linspace(-0.02, 0.02, 10)
            norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)

            # Plot smoothed MFC
            pc = ax.contourf(lon, lat, flux_convergence_smooth, extend='both',
                             levels=levels, cmap=cmap, norm=norm)

            # Colorbar
            divider = make_axes_locatable(ax)
            cbar = fig.colorbar(pc, extend='both', orientation="horizontal", fraction=0.06, pad=0.07)
            cbar.set_label('Moisture Flux Convergence [kg $m^{-2}$ $s^{-1}$]', fontsize=16)
            cbar.ax.tick_params(labelsize=16)
            cbar.set_ticks(levels)
            cbar.ax.set_xticklabels([f"{lvl:.3f}" for lvl in levels])

            # Plot 10m wind barbs (subsampled)
            step = 10
            ax.barbs(lons_np[::step, ::step], lats_np[::step, ::step],
                     u10_np[::step, ::step], v10_np[::step, ::step],
                     length=6, transform=ccrs.PlateCarree(),
                     pivot='middle', linewidth=1)

            # Save
            output_file = f"case01_100km_MFC_and_Wind_Barbs_700mb_{time_str}_LPF.jpg"
            fig.savefig(output_file, dpi=360, bbox_inches='tight')
            print(f"Saved plot to {output_file}")
            plt.close(fig)

    print("All plots generated successfully!")
