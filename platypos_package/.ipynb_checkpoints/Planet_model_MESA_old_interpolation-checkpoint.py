import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.interpolate import griddata
import os
import pandas as pd

# data to use for calculating the planetary radius in PLATYPOS
pathdir = os.getcwd().split("gitlab")[0]+'gitlab/mesagiants/make_planets_population1/'
smallgrid = "PLATYPOS_poulation1_5Rj_M_R_age_orbsep_10_to_5000Myr_interpM_small.csv"
biggrid = "PLATYPOS_poulation1_5Rj_M_R_age_orbsep_10_to_5000Myr_interpM_big.csv"

dfPLATYPOS_small = pd.read_csv(pathdir + smallgrid)
dfPLATYPOS_big = pd.read_csv(pathdir + biggrid)

####################################################################
# read in the data once, then only work with it
# grid of known radii and distances & R
# for all ages
a_R_age_points_small = []
M_age_values_small = []
a_M_age_points_small = []
R_age_values_small = []
for age_i in dfPLATYPOS_small.age.unique():
    for i in dfPLATYPOS_small.orb_sep[dfPLATYPOS_small.age == age_i].index:
        a = dfPLATYPOS_small.orb_sep[dfPLATYPOS_small.age == age_i].loc[i]
        for m in dfPLATYPOS_small.columns[2:].values:
            #
            R = dfPLATYPOS_small.loc[i][m]
            a_R_age_points_small.append([a, R, age_i])
            M_age_values_small.append(float(m))
            #
            a_M_age_points_small.append([a, float(m), age_i])
            R_age_values_small.append(dfPLATYPOS_small.loc[i][m])
            
a_R_age_points_small = np.array(a_R_age_points_small)
M_age_values_small = np.array(M_age_values_small)
a_M_age_points_small = np.array(a_M_age_points_small)
R_age_values_small = np.array(R_age_values_small)  

####################################################################
# read in the data once, then only work with it
# grid of known radii and distances & R
# for all ages
a_R_age_points_big = []
M_age_values_big = []
a_M_age_points_big = []
R_age_values_big = []
for age_i in dfPLATYPOS_big.age.unique():
    for i in dfPLATYPOS_big.orb_sep[dfPLATYPOS_big.age == age_i].index:
        a = dfPLATYPOS_big.orb_sep[dfPLATYPOS_big.age == age_i].loc[i]
        for m in dfPLATYPOS_big.columns[2:].values:
            #
            R = dfPLATYPOS_big.loc[i][m]
            a_R_age_points_big.append([a, R, age_i])
            M_age_values_big.append(float(m))
            #
            a_M_age_points_big.append([a, float(m), age_i])
            R_age_values_big.append(dfPLATYPOS_big.loc[i][m]) 
            
a_R_age_points_big = np.array(a_R_age_points_big)
M_age_values_big = np.array(M_age_values_big)
a_M_age_points_big = np.array(a_M_age_points_big)
R_age_values_big = np.array(R_age_values_big)  
####################################################################

def calculate_mass_planet_MESA(R_pl, a, age, grid):
    """
    Use MESA models to calculate planetary mass at a given radius, orbital separation and age
    NOTE: not all R_pl, a, age combinations might have a solution!
    NOTE: for now all planets have a 10 M_earth core with density 8 g/cm^3
    """
    R_pl_jup = R_pl/(const.R_jup/const.R_earth).value
    if grid == "small":
        a_R_age_points = a_R_age_points_small
        M_age_values = M_age_values_small
    elif grid == "big":
        a_R_age_points = a_R_age_points_big
        M_age_values = M_age_values_big

    # based on the underlying orbital separation-mass-age-radius grid I can 
    # interpolate within the grid to get any radius for a given combination of (a, M, age)
    # NOTE: for some radius, a, age combinations there might not be a solution
    
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

def calculate_radius_planet_MESA(M_pl, a, age, grid):
    """
    Use MESA models to calculate planetary radii at a given orbital separation, mass and age
    NOTE: for now all planets have a 10 M_earth core with density 8 g/cm^3
    """
    M_pl_jup = M_pl/(const.M_jup/const.M_earth).value
    
    if grid == "small":
        a_M_age_points = a_M_age_points_small
        R_age_values = R_age_values_small
    elif grid == "big":
        a_M_age_points = a_M_age_points_big
        R_age_values = R_age_values_big
    
    # based on the underlying orbital separation-mass-age-radius grid I can 
    # interpolate within the grid to get any radius for a given combination of (a, M, age)
    # NOTE: mass in [0.087, 11.3], a in [0.03, 1.0], age in [10, 10000]
    
    if (type(M_pl)==int) or (type(M_pl)==float) or (type(M_pl)==np.float64): # if M is single value
        point_i = (a, M_pl_jup, age)
        radius_i_jup = griddata(a_M_age_points, R_age_values, point_i, method='linear')
        return radius_i_jup * (const.R_jup/const.R_earth).value
        
    elif len(M_pl) > 1: # if M is array
        Rs_jup = []
        for i in range(len(M_pl)):
            point_i = (a[i], M_pl_jup[i], age[i])
            radius_i = griddata(a_M_age_points, R_age_values, point_i, method='linear')
            Rs.append(radius_i)
        Rs_jup = np.array(Rs_jup)
        return Rs_jup * (const.R_jup/const.R_earth).value
    
def density_planet(M_p, R_p):
    """once I have the radius and a mass estimate for the planet and can estimate a density"""  
    rho = (M_p*const.M_earth.cgs/(4./3*np.pi*(R_p*const.R_earth.cgs)**3)).cgs
    return rho.value