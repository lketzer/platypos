import numpy as np
import pandas as pd
from scipy import interpolate
import os
import platypos
import pkgutil
data = pkgutil.get_data(__package__, 'mamajeck_mainsequence.csv')
import io

def mass_lum_relation_mamajek():
    """ main-sequencen mass-luminosity relation from Erik Mamajek 
    (http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.dat)
    
    Returns:
    --------
    logL_from_M (function): interpolation function which takes a mass 
    value and returns the log10(bolometric luminosity)
    """ 

    df = pd.read_csv(io.StringIO(data.decode("utf-8")), sep="\s+")
    logL_from_M = interpolate.interp1d(df["Msun"], df["logL"], kind="linear")

    return logL_from_M
