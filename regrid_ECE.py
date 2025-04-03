
import os
import glob
import re
import numpy as np
import xarray as xr
import matplotlib.cbook as cbook
import pickle
import dask.array as da
import time 
import logging 
import pandas as pd
import yaml

from climtools import climtools_lib as ctl
from matplotlib import pyplot as plt
from scipy import stats
from smmregrid import Regridder, cdo_generate_weights
from cftime import DatetimeGregorian


# FROM UNSTRUCTURED TO LATLON GRID X EC-EARTH4 FILES
def regrid_files(input_pattern, target_grid, method, output_dir):
    """
    Regrids input files to a target grid using the smmregrid package.

    Parameters:
        input_pattern (str): Path pattern for input files (e.g., "/path/to/files/*.nc").
        target_grid (str): Target grid in CDO format (e.g., "r180x90").
        method (str): Interpolation method supported by CDO (e.g., "ycon").
        output_dir (str): Directory to save the regridded files.
    """
    # Use glob to match input files based on the pattern
    input_files = glob.glob(input_pattern)
    
    if not input_files:
        print(f"No files matched the pattern: {input_pattern}")
        return

    for file in input_files:
        # Load input dataset
        ds = xr.open_dataset(file)
        
        # Extract a sample of the source grid from the data
        source_grid = ds.isel(time_counter=0)  # Assuming 'time' is a dimension, time_counter for ECE4
        
        # Generate weights
        weights = cdo_generate_weights(source_grid, target_grid=target_grid, method=method)
        
        # Initialize Regridder with generated weights
        regridder = Regridder(weights=weights)
        
        # Apply regridding
        regridded_ds = regridder.regrid(ds)
        
        # Save the regridded file
        output_file = os.path.join(output_dir, f"{os.path.basename(file).replace('.nc', '_regridded.nc')}")
        regridded_ds.to_netcdf(output_file)
        print(f"Regridded file saved: {output_file}")
