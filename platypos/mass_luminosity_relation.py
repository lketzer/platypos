import numpy as np
import pandas as pd
from scipy import interpolate
import os
import platypos
#path_platypos = os.path.abspath(platypos.__file__).rstrip("__init__.py")[:-9]

def mass_lum_relation_mamajek():
    """ main-sequencen mass-luminosity relation from Erik Mamajek 
    (http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.dat)
    
    Returns:
    --------
    logL_from_M (function): interpolation function which takes a mass 
    value and returns the log10(bolometric luminosity)
    """ 

    df = pd.read_csv("../supplementary_files/Mamajek_MS.csv", sep="\s+")
    logL_from_M = interpolate.interp1d(df["Msun"], df["logL"], kind="linear")

    return logL_from_M
