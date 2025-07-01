import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
from pathlib import Path
import pandas as pd
from scipy.spatial import distance
import numpy as np

def load_variable(filepath):
    """Load a single variable from a NetCDF file"""
    ds = xr.open_dataset(filepath)
    # Escludi coordinate come 'time', 'lat', 'lon', 'lev'
    exclude_vars = {'time', 'time_counter', 'lat', 'lon', 'level', 'plev', 'time_bnds', 'time_counter_bnds'}
    physical_vars = [v for v in ds.data_vars if v not in exclude_vars]
    if not physical_vars:
        raise ValueError(f"No valid physical variables found in {filepath}")
    var = physical_vars[0]
    # Identify the correct time coordinate
    time_coord = None
    for candidate in ["time", "time_counter"]:
        if candidate in ds.coords or candidate in ds.variables:
            time_coord = candidate
            break
    # Decode CF-time if needed
    if time_coord and 'units' in ds[time_coord].attrs:
        ds[time_coord] = xr.decode_cf(ds[[time_coord]])[time_coord]
    # Optionally rename to have consistent naming
    if time_coord and time_coord != "time":
        ds = ds.rename({time_coord: "time"})

    return ds[var]

def compute_flux_from_files(files):
    """Compute Net TOA flux = rsdt - rsut - rlut"""
    flux_total = 0
    for f in files:
        da = load_variable(f).squeeze()
        if 'rsdt' in f:
            flux_total += da  # Incoming solar
        elif 'rsut' in f or 'rlut' in f:
            flux_total -= da  # Outgoing SW or LW
        else:
            raise ValueError(f"Unexpected variable in {f}. Expected 'rsdt', 'rsut', or 'rlut'.")
    return flux_total

def compute_gregory_slope(tsurf, flux, name="", plot=True, save_plot_dir=None,
                           outlier_method="mahalanobis", tsurf_threshold=285.0, show_all=True):
    data = np.vstack([tsurf, flux]).T

    # Detect outliers
    if outlier_method == "mahalanobis":
        mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        inv_cov = np.linalg.inv(cov)
        mdist = np.array([distance.mahalanobis(d, mean, inv_cov) for d in data])
        outliers = mdist > 1.8
    elif outlier_method == "low_tsurf":
        outliers = tsurf < tsurf_threshold
    else:
        outliers = np.full_like(tsurf, False, dtype=bool)

    # Linear regression on inliers only
    slope, intercept, r_value, p_value, std_err = linregress(tsurf[~outliers], flux[~outliers])
    x_intercept = -intercept / slope

    # Plotting
    if plot:
        plt.figure()
        if show_all:
            plt.scatter(tsurf[~outliers], flux[~outliers], label="Inliers", color="blue")
            plt.scatter(tsurf[outliers], flux[outliers], label="Outliers", color="red")
        else:
            plt.scatter(tsurf[~outliers], flux[~outliers], label="Inliers", color="blue")

        plt.plot(tsurf[~outliers],
                 intercept + slope * tsurf[~outliers],
                 color='green',
                 label=f'slope = {slope:.3f}')
        plt.xlabel("Tsurf [K]")
        plt.ylabel("Flux [W/m²]")
        plt.title(name)
        plt.legend()

        if save_plot_dir:
            Path(save_plot_dir).mkdir(exist_ok=True, parents=True)
            suffix = f"_{outlier_method}" if outlier_method else ""
            plt.savefig(f"{save_plot_dir}/{name}{suffix}.png")
        plt.close()

    return {
        "slope": slope,
        "intercept": intercept,
        "x_intercept": x_intercept,
        "r2": r_value**2,
        "std_err": std_err,
        "tsurf": tsurf,
        "flux": flux,
        "outliers": outliers
    }

def detect_simulation(label):
    # for sim i ["c4c9", "c49r", "c4C"]:
    for sim in ["pic9", "pi9r"]:
        if sim in label:
            return sim
    return "unknown"

def read_feedback_file(feedback_file):
    """Legge il file feedback e restituisce un dict {sezione: {feedback: valore, feedback_error: valore}}"""
    feedbacks = {}
    current = None
    with open(feedback_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.endswith("feedback:"):
                current = line[:-1].lower()  # esempio: "cld feedback"
                feedbacks[current] = {}
            elif ":" in line and current:
                key, val = line.split(":", 1)
                key = key.strip().replace(" ", "_").replace("-", "_").lower()
                try:
                    val = float(val.strip())
                except ValueError:
                    continue
                feedbacks[current][key] = val
    return feedbacks

def enrich_with_feedbacks(df, feedback_folder, feedback_names, label_col="label", group_col="group"):
    
    for fb in feedback_names:
        df[f"{fb}_fb"] = pd.NA
        df[f"{fb}_err"] = pd.NA
        df[f"{fb}_cs_fb"] = pd.NA   # aggiunto per clear sky
        df[f"{fb}_cs_err"] = pd.NA
    
    for idx, row in df.iterrows():
        model_id = row[group_col]
        feedback_file = os.path.join(feedback_folder, f"feedbacks_{model_id}")
        if not os.path.exists(feedback_file):
            print(f"Warning: {feedback_file} not found.")
            continue
        
        fb_data = read_feedback_file(feedback_file)
        
        for fb in feedback_names:
            fb_key = fb.replace("-", "_")
            # cld feedback
            fb_val_cld = fb_data.get("cld feedback", {}).get(f"{fb_key}_feedback")
            fb_err_cld = fb_data.get("cld feedback", {}).get(f"{fb_key}_feedback_error")
            # clr feedback
            fb_val_clr = fb_data.get("clr feedback", {}).get(f"{fb_key}_feedback")
            fb_err_clr = fb_data.get("clr feedback", {}).get(f"{fb_key}_feedback_error")
            
            # assegna in base al suffisso _CS nel label
            if "_CS" in row[label_col]:
                if fb_val_clr is not None:
                    df.at[idx, f"{fb}_cs_fb"] = fb_val_clr
                if fb_err_clr is not None:
                    df.at[idx, f"{fb}_cs_err"] = fb_err_clr
            else:
                if fb_val_cld is not None:
                    df.at[idx, f"{fb}_fb"] = fb_val_cld
                if fb_err_cld is not None:
                    df.at[idx, f"{fb}_err"] = fb_err_cld

    return df

def plot_grouped_gregory(df, flux_group, save_dir=None, show_outliers=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = {"pic9": "tab:blue", "pi9r": "tab:orange", "piC": "tab:green", "unknown": "grey"}
    #palette = {"c4c9": "tab:blue", "c49r": "tab:orange", "c4C": "tab:green", "unknown": "grey"}
    marker_map = {"": "o", "_CS": "s"}  # Cerchi per all-sky, quadrati per CS
    linestyle_map = {"": "-", "_CS": "--"}

    # Creare legenda custom per marker
    legend_handles = []

    for flux in flux_group:
        subset = df[df["label"].str.endswith(flux)]
        for label in subset["label"].unique():
            row = subset[subset["label"] == label].iloc[0]
            tsurf = row["tsurf"]
            flux_vals = row["flux"]
            outliers = row.get("outliers", [])

            base_label = detect_simulation(label)
            is_CS = "CS" in label

            color = palette[base_label]
            marker = marker_map["CS" if is_CS else ""]
            linestyle = linestyle_map["_CS" if is_CS else ""]

            label_display = f"{base_label}{'_CS' if is_CS else ''} ({flux}): slope = {row['slope']:.2f}"

            # Scatter dei punti
            outliers = np.array(row.get("outliers", []), dtype=bool)
            if show_outliers or not outliers.any():
                ax.scatter(tsurf, flux_vals, label=label_display, color=color, marker=marker, alpha=0.7)
            else:
                mask = ~outliers
                ax.scatter(np.array(tsurf)[mask], np.array(flux_vals)[mask], label=label_display, color=color, marker=marker, alpha=0.7)
            
            tsurf = np.array(tsurf).flatten()
            flux_vals = np.array(flux_vals).flatten()
            # Regressione
            m, b = row["slope"], row["intercept"]
            ts_range = np.linspace(np.min(tsurf), np.max(tsurf), 100)
            ax.plot(ts_range, m * ts_range + b, color=color, linestyle=linestyle)

            if "albedo_fb" in row and not pd.isna(row["albedo_fb"]):
                mean_t = np.mean(tsurf)
                ax.errorbar(mean_t, row["albedo_fb"], yerr=row["albedo_err"], fmt="D", color=color, alpha=0.8, label=f"Albedo ({label}): {row['albedo_fb']:.2f}")
        
    # Legenda e asse
    ax.set_title(f"Gregory Comparison: {flux_group[0]} vs {flux_group[1]}")
    ax.set_xlabel("Tsurf [K]")
    ax.set_ylabel("Flux [W/m²]")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True)

    if save_dir:
        fname = "_vs_".join(flux_group) + ".png"
        full_path = os.path.join(save_dir, fname)
        plt.savefig(full_path, dpi=300)
        print(f"Plot salvato in: {full_path}")

    plt.show()

def process_simulation(ts_file, flux_file_or_list, label, save_plot_dir=None, outlier_method="mahalanobis", tsurf_threshold=1.0, show_all=True):
    ts = load_variable(ts_file).squeeze()

    if isinstance(flux_file_or_list, str):
        flux = compute_flux_from_files([flux_file_or_list])
    elif isinstance(flux_file_or_list, list):
        flux = compute_flux_from_files(flux_file_or_list)
    else:
        raise ValueError("flux_file_or_list must be a string or list of strings")

    result = compute_gregory_slope(
        ts, flux,
        name=label,
        plot=True,
        save_plot_dir=save_plot_dir,
        outlier_method=outlier_method,
        tsurf_threshold=tsurf_threshold,
        show_all=show_all
    )

    return {
        "label": label,
        "slope": result["slope"],
        "intercept": result["intercept"],
        "x_intercept": result["x_intercept"],
        "r2": result["r2"],
        "std_err": result["std_err"],
        "tsurf": result["tsurf"],
        "flux": result["flux"],
        "outliers": result["outliers"],
        "group": label.split("_")[0],  # e.g. pi9r or pic9
        "flux_type": label.split("_", 1)[1]  # e.g. Net_TOA
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
        outlier_method = pair.get("outlier_method", "mahalanobis")  # Default to mahalanobis
        tsurf_threshold = pair.get("tsurf_threshold", 289)          # Used only if outlier_method == "low_tsurf"
        show_all = pair.get("show_all", True)  # default True
        result = process_simulation(ts_path, flux_paths, label, save_plot_dir, outlier_method, tsurf_threshold, show_all)
        results.append(result)

    return pd.DataFrame(results)