import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.interpolate import griddata
from scipy import interpolate
import os
import pandas as pd


def initialize_R_interpolation_MESA_grid():
    """ Call this function at the beginning of PLATYPOS run!
    Initialize the radius-interpolation functions, which are 
    used to get a radius for a (age, mass, orb_sep) point"""
    
    # data to use for calculating the planetary radius in PLATYPOS
    ###########
    # outdated!
    # pathdir = os.getcwd().split("gitlab")[0]+'gitlab/mesagiants/make_planets_population1/'
    # grid = "PLATYPOS_poulation1_5Rj_M_R_age_orbsep_5_to_5000Myr_interpM_big.csv"
    ###########
    
    # I have MESA models with 0, 10, and 25 Mearth mass cores
    pathdir = os.getcwd().split("gitlab")[0]+'gitlab/mesagiants/Results_population4_2/Tables/'
    grid_Mcore0 = "PLATYPOS_poulation4_2_diffRstart_M_R_age_orbsep_5_to_6000Myr_Mcore0_origM_dense.csv"
    grid_Mcore10 = "PLATYPOS_poulation4_2_diffRstart_M_R_age_orbsep_5_to_6000Myr_Mcore10_origM_dense.csv"
    grid_Mcore25 = "PLATYPOS_poulation4_2_diffRstart_M_R_age_orbsep_5_to_6000Myr_Mcore25_origM_dense.csv"
    # drop columns which have NaN entries, because interpolate.RectBivariateSpline can't handle it!!
    dfPLATYPOS_Mcore0 = pd.read_csv(pathdir + grid_Mcore0).dropna(axis='columns')
    dfPLATYPOS_Mcore10 = pd.read_csv(pathdir + grid_Mcore10).dropna(axis='columns')
    dfPLATYPOS_Mcore25 = pd.read_csv(pathdir + grid_Mcore25).dropna(axis='columns')
    dict_dfPLATYPOS = {"Mcore0": dfPLATYPOS_Mcore0, 
                       "Mcore10": dfPLATYPOS_Mcore10, 
                       "Mcore25": dfPLATYPOS_Mcore25}

    ####################################################################
    # interpolate once at beginning to create function -> then use this function in PLATYPOS
    # then I have an interpolating function defined for each orbital separation
    # just need to plug in the current point (age, mass core, mass, orb_sep)
    # into the appropriate interp.-function to get corresponding current radius
    # NOTE: this is MUCH faster than interpolating in the table every single time step!
    
    # these are the functions to get a radius
    interp2d_funct_R_for_each_a_Mcore0 = {}
    interp2d_funct_R_for_each_a_Mcore10 = {}
    interp2d_funct_R_for_each_a_Mcore25 = {}
    dict_interp2d_funct = {"Mcore0": interp2d_funct_R_for_each_a_Mcore0, 
                           "Mcore10": interp2d_funct_R_for_each_a_Mcore10, 
                           "Mcore25": interp2d_funct_R_for_each_a_Mcore25}
    
    for Mcore in ["Mcore0", "Mcore10", "Mcore25"]:
        dfPLATYPOS = dict_dfPLATYPOS[Mcore]
        interp2d_funct_R_for_each_a = dict_interp2d_funct[Mcore]
        for a in dfPLATYPOS.orb_sep.unique():
            df_a = dfPLATYPOS[dfPLATYPOS.orb_sep == a] # dataframe for one orbital separation
            age_arr = df_a.age.values
            mass_arr = df_a.columns[2:].astype(float).values
            R_age_a_i = []
            for age_key in df_a.index:
                R_mass = []
                for mass in df_a.columns[2:]:
                    R = df_a[mass].loc[age_key]
                    R_mass.append(R)
                R_age_a_i.append(R_mass)
            # now for each orb-sep, I have an array with len(age_arr) rows, where each row has len(mass_arr) 
            # entries of the corresponding radius to the row age and column mass
            R_age_a_i = np.array(R_age_a_i) 
            # R_age_a_i is the 2-D input for scipy.interpolate.RectBivariateSpline
            f_a = interpolate.RectBivariateSpline(age_arr, mass_arr, R_age_a_i)
            # save this function in dictionary (one for each orbital separation)
            # now for each orbital separation a I can interpolate within an age-mass-radius grid using the returned interp. function
            interp2d_funct_R_for_each_a[str(a)] = f_a
    
    return dict_interp2d_funct


def calculate_radius_planet_MESA(M_pl, Mcore, a, age, interp_R_dict):
    """ 
    Use MESA models to calculate planetary radius at a given radius, orbital separation and age.
    
    Parameters:
    ----------
    M_pl: planetary mass [input in Earth masses, but grid in M_jup]
    Mcore: planetary core mass [input in Earth masses]
    a: semi-major axis [needs to be a value from MESA-model-grid]
    age: planetary age [needs to be a value from within the MESA-model-grid age boundaries]
    grid: dictionary with radius-interpolation functions for each 
          semi-major axis in MESA-model-grid (interp2d_funct_R_for_each_a)
    ----------
    
    NOTE: not all R_pl, a, age combinations might have a solution!
    NOTE: for now all planets have a 10 M_earth core with density 8 g/cm^3
    """
    Mcore = "Mcore"+str(int(float(Mcore)))
    interp_R_dict = interp_R_dict[Mcore]
    
    M_pl_jup = M_pl/(const.M_jup/const.M_earth).value
    R = interp_R_dict[str(a)](age, M_pl_jup)[:,0]
    if len(R)==1:
        return R[0]*(const.R_jup/const.R_earth).value # return in Earth radii
    else:
        return R*(const.R_jup/const.R_earth).value


def calculate_mass_planet_MESA(R_pl, Mcore, a, age):
    """   
    THIS IS NOT FINALIZED! USE WITH CARE...
    
    Use MESA models to calculate planetary mass at a given mass, orbital separation and age
    
    Parameters:
    ----------
    R_pl: planetary radius [input in Earth radii, but grid in R_jup ]
    a: semi-major axis [needs to be a value from MESA-model-grid]
    age: planetary age [needs to be a value from within the MESA-model-grid age boundaries]
    grid: dictionary with radius-interpolation functions for each 
          semi-major axis in MESA-model-grid (interp2d_funct_M_for_each_a)
    ----------
    
    NOTE: not all M_pl, a, age combinations might have a solution! (in particular when chosen mass is off the grid)
    NOTE: for now all planets have a 10 M_earth core with density 8 g/cm^3
    """
    
    pathdir = os.getcwd().split("gitlab")[0]+'gitlab/mesagiants//Results_population4_2/Tables/'
    grid_Mcore0 = "PLATYPOS_poulation4_2_diffRstart_M_R_age_orbsep_5_to_6000Myr_Mcore0_origM_dense.csv"
    grid_Mcore10 = "PLATYPOS_poulation4_2_diffRstart_M_R_age_orbsep_5_to_6000Myr_Mcore10_origM_dense.csv"
    grid_Mcore25 = "PLATYPOS_poulation4_2_diffRstart_M_R_age_orbsep_5_to_6000Myr_Mcore25_origM_dense.csv"
    dfPLATYPOS_Mcore0 = pd.read_csv(pathdir + grid_Mcore0)
    dfPLATYPOS_Mcore10 = pd.read_csv(pathdir + grid_Mcore10)
    dfPLATYPOS_Mcore25 = pd.read_csv(pathdir + grid_Mcore25)
    dict_dfPLATYPOS = {"Mcore0": dfPLATYPOS_Mcore0, 
                       "Mcore10": dfPLATYPOS_Mcore10, 
                       "Mcore25": dfPLATYPOS_Mcore25}
    
    Mcore = "Mcore"+str(int(float(Mcore)))
    dfPLATYPOS = dict_dfPLATYPOS[Mcore]
    
    # this step takes some time; USE WISELY
    ####################################################################
    # read in the data once, then only work with it
    # grid of known radii and distances & R
    # for all ages
    a_R_age_points = []
    M_age_values = []
    for age_i in dfPLATYPOS.age.unique():
        for i in dfPLATYPOS.orb_sep[dfPLATYPOS.age == age_i].index:
            a = dfPLATYPOS.orb_sep[dfPLATYPOS.age == age_i].loc[i]
            for m in dfPLATYPOS.columns[2:].values:
                R = dfPLATYPOS.loc[i][m]
                a_R_age_points.append([a, R, age_i])
                M_age_values.append(float(m))

    a_R_age_points = np.array(a_R_age_points)
    M_age_values = np.array(M_age_values)
    ####################################################################
    
    R_pl_jup = R_pl/(const.R_jup/const.R_earth).value
    if (type(R_pl)==int) or (type(R_pl)==float) or (type(R_pl)==np.float64): # if R is single value
        point_i = (a, R_pl_jup, age)
        mass_i_jup = griddata(a_R_age_points, M_age_values, point_i, method='linear')
        return mass_i_jup * (const.M_jup/const.M_earth).value

    elif len(R_pl) > 1: # if R is array
        Ms = []
        for i in range(len(R_pl)):
            point_i = (a[i], R_pl_jup[i], age[i])
            mass_i = griddata(a_R_age_points, M_age_values, point_i, method='linear')
            Ms.append(mass_i)
        Ms_jup = np.array(Ms_jup)
        return Ms_jup * (const.M_jup/const.M_earth).value
    
def density_planet(M_p, R_p):
    """once I have the radius and a mass estimate for the planet and can estimate a density (in cgs units)"""  
    rho = (M_p*const.M_earth.cgs/(4./3*np.pi*(R_p*const.R_earth.cgs)**3)).cgs
    return rho.value