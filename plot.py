
import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import re

##ECE4
def load_dRt_ece4(base_folder, feedbacks, param_name, valchange, subfolder_pattern="s00*"):
    """Load dRt values for ECE4 from NetCDF files in variation subfolders."""
    if param_name not in valchange:
        print(f"Parameter {param_name} not found in valchange.")
        return None, None

    variations = valchange[param_name]

    # Mapping last digit to percentage values
    variation_map = {
        "1": -20,
        "2": 20,
        "3": 30
    }

    feedback_data = {fb: [] for fb in feedbacks}
    variation_values = {fb: [] for fb in feedbacks}

    for subfolder in glob.glob(os.path.join(base_folder, subfolder_pattern)):
        if not os.path.isdir(subfolder) or len(os.path.basename(subfolder)) < 4:
            continue

        # Extract the last character of the folder name
        variation_key = os.path.basename(subfolder)[-1]  

        variation = variation_map.get(variation_key)
        if variation is None:
            print(f"Variation '{variation_key}' not recognized in {subfolder}, skipped.")
            continue

        for fb in feedbacks:
            file_path = os.path.join(subfolder, fb)
            if "dRt" not in fb or not fb.endswith(".nc"):
                continue

            try:
                ds = xr.open_dataset(file_path)
                if "__xarray_dataarray_variable__" in ds.data_vars:
                    dRt_mean = float(ds["__xarray_dataarray_variable__"].values)
                    feedback_data[fb].append(dRt_mean)
                    variation_values[fb].append(variation)
            except Exception as e:
                print(f"Error in reading {file_path}: {e}")

    return feedback_data, variation_values

def plot_dRt_ece4(feedback_data, variation_values, labels, xlabel, ylabel, title, output_file=None):
    """Plot dRt values against parameter variations for ECE4."""
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, (fb, label) in enumerate(zip(feedback_data.keys(), labels)):
        if feedback_data[fb]:
            plt.scatter(variation_values[fb], feedback_data[fb], color=colors[i % len(colors)], label=label, s=100)

    plt.xticks([-30, -20, -10, 0, 10, 20, 30])
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold', style='italic')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved in {output_file}")
    else:
        plt.show()

##ECE 3
def load_dRt_ece3(base_folder, feedbacks, param_name, valchange, subfolder_pattern="pi*a"):
    """dRt value upload from NetCDF file in variation corrispondent subfolder."""
    if param_name not in valchange:
        print(f"Parameter {param_name} not found in valchange.")
        return None, None

    variations = valchange[param_name]
    variation_keys = ['n', 'l', 'r', 'p']
    variation_map = dict(zip(variation_keys, variations))

    feedback_data = {fb: [] for fb in feedbacks}
    variation_values = {fb: [] for fb in feedbacks}

    for subfolder in glob.glob(os.path.join(base_folder, subfolder_pattern)):
        if not os.path.isdir(subfolder) or len(os.path.basename(subfolder)) < 3: # Check if the folder name is too short
            continue

        variation_key = os.path.basename(subfolder)[2] # Extract variation letter (e.g., 'm', 'n', etc.)
        variation = variation_map.get(variation_key)

        if variation is None:
            print(f"Variation '{variation_key}' not found {subfolder}, skipped.")
            continue

        for fb in feedbacks:
            file_path = os.path.join(subfolder, fb)
            if "dRt" not in fb or not fb.endswith(".nc"):
                continue

            try:
                ds = xr.open_dataset(file_path)
                if "__xarray_dataarray_variable__" in ds.data_vars:
                    dRt_mean = float(ds["__xarray_dataarray_variable__"].values)
                    feedback_data[fb].append(dRt_mean)
                    variation_values[fb].append(variation)
            except Exception as e:
                print(f"Error in reading {file_path}: {e}")

    return feedback_data, variation_values

def plot_dRt_ece3(feedback_data, variation_values, labels, xlabel, ylabel, title, output_file=None, xtick_labels=None):
    """dRt plot with variation"""
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, (fb, label) in enumerate(zip(feedback_data.keys(), labels)):
        if feedback_data[fb]:
            plt.scatter(variation_values[fb], feedback_data[fb], color=colors[i % len(colors)], label=label, s=100)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold', style='italic')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(variation_values[fb], rotation=45,fontsize=10)
    plt.yticks(fontsize=10)

    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot salvato in {output_file}")
    else:
        plt.show()

##ALL param
def plot_dRt_fb_all_params_norm(base_folder, valchange, xlabel, ylabel, title, output_file=None, subfolder_pattern="pip*", model_type="ece3"):
    """
    Plot normalized dRt values for different parameters with separate mappings for ECE3 and ECE4.
    
    Parameters:
    -----------
    base_folder : str
        The base folder containing the subfolders with the data files.
    valchange : dict
        Dictionary containing sensitivity values for each parameter.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title : str
        Title for the plot.
    output_file : str, optional
        File path to save the plot (default: None, plot shown interactively).
    subfolder_pattern : str, optional
        Subfolder pattern to match files (default: "pip*").
    model_type : str, optional
        Specify the model type ("ece3" or "ece4").
    """
    feedback_file = "dRt_planck-surf_global_clr_climatology-HUANGkernels.nc"
    color_map = plt.cm.get_cmap("tab10")
    
    param_names = []
    sensitivities = []
    dRt_values = []

    # Define parameter-to-name mapping flexibly
    param_to_real_name = {
        "ece3": {
            "pi*a": "ENTRORG", 
            "pi*b": "RPRCON", 
            "pi*c": "DETRPEN", 
            "pi*d": "RMFDEPS", 
            "pi*e": "RVICE", 
            "pi*f": "RSNOWLIN2", 
            "pi*g": "RCLDIFF", 
            "pi*h": "RLCRIT_UPHYS"
        },
        "ece4": {
            "s00*": "RPRCON", 
            "s01*": "ENTRORG", 
            "s02*": "DETRPEN", 
            "s03*": "ENTRDD", 
            "s04*": "RMFDEPS", 
            "s05*": "RVICE", 
            "s06*": "RLCRITSNOW", 
            "s07*": "RSNOWLIN2"
        }
    }
    
    if model_type not in param_to_real_name:
        print("Invalid model type specified. Please choose 'ece3' or 'ece4'.")
        return
    
    subfolders = glob.glob(os.path.join(base_folder, subfolder_pattern))
    
    for subfolder_path in subfolders:
        if os.path.isdir(subfolder_path):
            param_name = os.path.basename(subfolder_path)  
            
            matched_param = None
            for pattern, real_name in param_to_real_name[model_type].items():
                if re.match(pattern.replace('*', '.*'), param_name):
                    matched_param = real_name
                    break
            
            if matched_param is None:
                print(f"Skipping unknown subfolder {param_name}")
                continue
            
            file_path = os.path.join(subfolder_path, feedback_file)
            if not os.path.exists(file_path):
                print(f"File {file_path} not found, skipping.")
                continue
            
            try:
                ds = xr.open_dataset(file_path)
                if "__xarray_dataarray_variable__" in ds.data_vars:
                    dRt_mean = float(ds["__xarray_dataarray_variable__"].values)
                    param_names.append(matched_param)
                    
                    raw_s_value = valchange.get(matched_param, [])
                    if raw_s_value:
                        if model_type == "ece3":
                            sensitivity_value = raw_s_value[-1]  
                            norm_sensitivity = sensitivity_value / max(abs(np.array(raw_s_value)))  
                        elif model_type == "ece4":
                            sensitivity_value = raw_s_value[-1]  
                            norm_sensitivity = sensitivity_value / 100  
                        sensitivities.append(norm_sensitivity)
                        dRt_values.append(dRt_mean)
                    else:
                        print(f"Warning: No sensitivity value for {matched_param}, skipping.")
                else:
                    print(f"File {file_path} missing '__xarray_dataarray_variable__'. Skipping...")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    if len(sensitivities) != len(dRt_values):
        print("Warning: Mismatch in the number of sensitivities and dRt values!")
        return

    plt.figure(figsize=(8, 6))

    for i, param_name in enumerate(param_names):
        color = color_map(i % 10)  
        plt.scatter(sensitivities[i], dRt_values[i], color=color, s=100)

        plt.annotate(param_name, 
                    (sensitivities[i], dRt_values[i]),
                    fontsize=10, ha='right', color=color,
                    xytext=(10, 10),  
                    textcoords='offset points')  

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold', style='italic')  
    plt.grid(False)

    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def plot_dRt_fb_all_params(base_folder, xlabel, ylabel, title, feedback_file, output_file=None, subfolder_pattern="pip*", model_type="ece3"):
    """
    Plot dRt values for different parameters with separate mappings for ECE3 and ECE4.

    Parameters:
    -----------
    base_folder : str
        The base folder containing the subfolders with the data files.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title : str
        Title for the plot.
    feedback_file : str
        The specific feedback file to use (e.g., "dRt_planck-surf_global_clr_climatology-HUANGkernels.nc").
    output_file : str, optional
        File path to save the plot (default: None, plot shown interactively).
    subfolder_pattern : str, optional
        Subfolder pattern to match files (default: "pip*").
    model_type : str, optional
        Specify the model type ("ece3" or "ece4").
    """
    
    param_names = []
    dRt_values = []
    
    subfolders = glob.glob(os.path.join(base_folder, subfolder_pattern))

    # Define parameter name mapping for ECE3 and ECE4
    if model_type == "ece3":
        param_to_real_name = {
            "pi*a": "ENTRORG", 
            "pi*b": "RPRCON", 
            "pi*c": "DETRPEN", 
            "pi*d": "RMFDEPS", 
            "pi*e": "RVICE", 
            "pi*f": "RSNOWLIN2", 
            "pi*g": "RCLDIFF", 
            "pi*h": "RLCRIT_UPHYS"
        }
    elif model_type == "ece4":
        param_to_real_name = {
            "s00*": "RPRCON", 
            "s01*": "ENTRORG", 
            "s02*": "DETRPEN", 
            "s03*": "ENTRDD", 
            "s04*": "RMFDEPS", 
            "s05*": "RVICE", 
            "s06*": "RLCRITSNOW", 
            "s07*": "RSNOWLIN2"
        }
    else:
        print("Invalid model type specified. Please choose 'ece3' or 'ece4'.")
        return
    
    for subfolder_path in subfolders:
        if os.path.isdir(subfolder_path):
            subfolder_name = os.path.basename(subfolder_path)  # Get the parameter name (e.g., "pipa", "pipb", etc.)
            # Match the parameter pattern based on model type
            matched_param = None
            for pattern, real_name in param_to_real_name.items():
                if re.match(pattern.replace('*', r'\w'), subfolder_name):  # Replace '*' with wildcard for matching
                    matched_param = real_name
                    break
            if matched_param is None:
                print(f"Skipping unknown subfolder {subfolder_name}")
                continue
            
            # Check if the required feedback file exists in the subfolder
            file_path = os.path.join(subfolder_path, feedback_file)
            if not os.path.exists(file_path):
                print(f"File {file_path} not found, skipping.")
                continue
            
            try:
                ds = xr.open_dataset(file_path)
                if "__xarray_dataarray_variable__" in ds.data_vars:
                    dRt_mean = float(ds["__xarray_dataarray_variable__"].values)
                    param_names.append(matched_param)  
                    dRt_values.append(dRt_mean)
                else:
                    print(f"File {file_path} missing '__xarray_dataarray_variable__'. Skipping...")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Plot the results
    num_params = len(param_names)
    if num_params == 0:
        print("No data to plot.")
        return
    color_map = plt.cm.get_cmap("tab10")
    
    plt.figure(figsize=(8, 6))
    for i, param_name in enumerate(param_names):
        plt.scatter(i, dRt_values[i], color=color_map(i % num_params), s=100)  # Point with color
    
    plt.xticks(range(num_params), param_names, rotation=45, ha="right", color='black')
    for i, label in enumerate(plt.gca().get_xticklabels()):
        label.set_color(color_map(i % num_params))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontweight='bold', fontsize=12)
    plt.grid(False)  # Remove grid
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
    plt.show()

## TIME SERIES
def plot_toa_anomaly_ece3(base_folder, param, dRt_folder, title, output_file=None):
    """
    Plots the TOA anomaly and dRt components for a given ECE3 parameter.
    
    Parameters:
    - base_folder: str, base path containing the simulation and control datasets
    - param: str, parameter identifier (e.g., 'pipb')
    - dRt_folder: str, folder containing dRt component files
    - title: str, title for the plot
    - output_file: str, optional, file path to save the plot
    """
    # Load simulation and control datasets
    sim_path = os.path.join(base_folder, "pi", "t_sim", param, f"{param}_*_tnr.nc")
    ctrl_path = os.path.join(base_folder, "pi", "std_sim", "tpa1", "tpa1_*_tnr.nc")
    
    ds_sim = xr.open_mfdataset(sim_path, combine="by_coords")
    ds_ctrl = xr.open_mfdataset(ctrl_path, combine="by_coords")
    
    # Compute annual mean anomaly
    tnr_sim_annual = ds_sim["tnr"].groupby("time.year").mean(dim="time")
    tnr_ctrl_annual = ds_ctrl["tnr"].groupby("time.year").mean(dim="time")
    climatology = tnr_ctrl_annual.mean(dim="year")
    tnr_anomaly = (tnr_sim_annual - climatology).mean(dim=["lat", "lon"]).to_pandas()
    
    # Load and sum dRt components
    dRt_files = [
        "dRt_albedo_global_cld_climatology-HUANGkernels.nc",
        "dRt_lapse-rate_global_cld_climatology-HUANGkernels.nc",
        "dRt_planck-atmo_global_cld_climatology-HUANGkernels.nc",
        "dRt_planck-surf_global_cld_climatology-HUANGkernels.nc",
        "dRt_water-vapor_global_cld_climatology-HUANGkernels.nc",
    ]
    
    dRt_components_pd = [xr.open_dataset(os.path.join(dRt_folder, f))["__xarray_dataarray_variable__"].to_pandas() for f in dRt_files]
    dRt_sum_pd = sum(dRt_components_pd)
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(tnr_anomaly.index, tnr_anomaly, color="black", marker="o", linestyle="-", label="Net TOA Anomaly")
    
    colors = ["blue", "green", "purple", "orange", "cyan"]
    labels = ["Albedo", "Lapse Rate", "Planck Atmos", "Planck Surface", "Water Vapor"]
    
    for comp_pd, color, label in zip(dRt_components_pd, colors, labels):
        plt.plot(comp_pd.index, comp_pd, color=color, marker="s", linestyle="--", label=label)
    
    plt.plot(dRt_sum_pd.index, dRt_sum_pd, color="red", linestyle="-", linewidth=2, label="Sum of dRt Components")
    
    plt.xlabel("Year")
    plt.ylabel("W/m²")
    plt.title(title, fontsize=14)
    plt.axhline(0, color="gray", linestyle="--")
    plt.legend()
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_toa_anomaly_ece4(base_folder, param, dRt_folder, title, output_file=None):
    """
    Plots the TOA anomaly (calculated from rsntcs - rlntcs) and dRt components.
    
    Parameters:
    - base_folder: str, base path containing the simulation and control datasets
    - param: str, parameter identifier (e.g., 's001')
    - dRt_folder: str, folder containing dRt component files
    - title: str, title for the plot
    - output_file: str, optional, file path to save the plot
    """
    
    def calculate_tnrcs(file_pattern):
        """Helper function to calculate tnrcs from rsntcs and rlntcs."""
        ds = xr.open_mfdataset(file_pattern, combine="by_coords")
        tnrcs = ds["rsntcs"] - ds["rlntcs"]
        return tnrcs
    
    # Construct file paths for simulation and control data
    sim_path = os.path.join(base_folder, "t_sim", param, "oifs", "regridded", f"{param}_*_1m_*.nc")
    ctrl_path = os.path.join(base_folder, "std_sim", "oifs", f"s000_*_1m_*.nc") #this path has been changed.

    # Calculate tnrcs for simulation and control
    tnrcs_sim = calculate_tnrcs(sim_path)
    tnrcs_ctrl = calculate_tnrcs(ctrl_path)
    
    # Compute annual mean anomaly
    tnr_sim_annual = tnrcs_sim.groupby("time_counter.year").mean(dim="time_counter")
    tnr_ctrl_annual = tnrcs_ctrl.groupby("time_counter.year").mean(dim="time_counter")
    climatology = tnr_ctrl_annual.mean(dim="year")
    tnr_anomaly = (tnr_sim_annual - climatology).mean(dim=["lat", "lon"]).to_pandas()
    
    # Load and sum dRt components
    dRt_files = [
        "dRt_lapse-rate_global_cld_climatology-HUANGkernels.nc",
        "dRt_planck-atmo_global_cld_climatology-HUANGkernels.nc",
        "dRt_planck-surf_global_cld_climatology-HUANGkernels.nc",
        "dRt_water-vapor_global_cld_climatology-HUANGkernels.nc",
    ]
    
    dRt_components_pd = [xr.open_dataset(os.path.join(dRt_folder, f))["__xarray_dataarray_variable__"].to_pandas() for f in dRt_files]
    dRt_sum_pd = sum(dRt_components_pd)
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(tnr_anomaly.index, tnr_anomaly, color="black", marker="o", linestyle="-", label="Net TOA Anomaly")
    
    colors = ["blue", "green", "purple", "orange"]
    labels = ["Lapse Rate", "Planck Atmos", "Planck Surface", "Water Vapor"]
    
    for comp_pd, color, label in zip(dRt_components_pd, colors, labels):
        plt.plot(comp_pd.index, comp_pd, color=color, marker="s", linestyle="--", label=label)
    
    plt.plot(dRt_sum_pd.index, dRt_sum_pd, color="red", linestyle="-", linewidth=2, label="Sum of dRt Components")
    
    plt.xlabel("Year")
    plt.ylabel("W/m²")
    plt.title(title, fontsize=14)
    plt.axhline(0, color="gray", linestyle="--")
    plt.legend()
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()