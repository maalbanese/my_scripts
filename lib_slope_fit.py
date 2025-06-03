import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
from pathlib import Path
import pandas as pd

def load_variable(filepath):
    """Load a single variable from a NetCDF file"""
    ds = xr.open_dataset(filepath)
    var = list(ds.data_vars)[0]
    return ds[var]

def compute_flux_from_files(files):
    """Compute combined flux from list of files, correcting signs if needed"""
    flux_total = 0
    for f in files:
        da = load_variable(f).squeeze()
        if any(key in f for key in ['rlut', 'rsut']):
            da = -da  # outgoing -> negative for energy budget
        flux_total += da
    return flux_total

def compute_gregory_slope(tsurf, flux, name="", plot=True, save_plot_dir=None):
    # Compute anomalies with respect to the first value
    tsurf_anom = tsurf - tsurf[0]
    flux_anom = flux - flux[0]
    slope, intercept, r_value, _, _ = linregress(tsurf_anom, flux_anom)
    if plot:
        plt.figure()
        plt.scatter(tsurf_anom, flux_anom, label="data")
        plt.plot(tsurf_anom, intercept + slope * tsurf_anom, color='r', label=f'slope={slope:.3f}')
        plt.xlabel("ΔTsurf [K]")
        plt.ylabel("ΔFlux [W/m²]")
        plt.title(name)
        plt.legend()
        if save_plot_dir:
            Path(save_plot_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig(f"{save_plot_dir}/{name}.png")
        plt.close()

    return slope, intercept, r_value**2

def process_simulation(ts_file, flux_file_or_list, label, save_plot_dir=None):
    ts = load_variable(ts_file).squeeze()

    # Normalize flux to a DataArray (sum if list, correct signs)
    if isinstance(flux_file_or_list, str):
        flux = compute_flux_from_files([flux_file_or_list])
    elif isinstance(flux_file_or_list, list):
        flux = compute_flux_from_files(flux_file_or_list)
    else:
        raise ValueError("flux_file_or_list must be a string or list of strings")

    slope, intercept, r2 = compute_gregory_slope(ts, flux, name=label, plot=True, save_plot_dir=save_plot_dir)

    return {
        "label": label,
        "slope": slope,
        "intercept": intercept,
        "r2": r2
    }

def batch_process(folder, var_pairs, save_plot_dir=None):
    results = []
    for pair in var_pairs:
        ts_path = os.path.join(folder, pair['ts'])
        flux_entry = pair['flux']
        if isinstance(flux_entry, list):
            flux_paths = [os.path.join(folder, f) for f in flux_entry]
        else:
            flux_paths = os.path.join(folder, flux_entry)

        label = pair['label']
        result = process_simulation(ts_path, flux_paths, label, save_plot_dir)
        results.append(result)
    return pd.DataFrame(results)