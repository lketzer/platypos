# Lopez & Fortney 2014 models
import astropy.units as u
from astropy import constants as const
import numpy as np

def calculate_core_radius(M_core):
    """ M-R relation for rock/iron Earth-like core. (no envelope) """
    R_core = (M_core**0.25)
    return R_core

def calculate_planet_mass(M_core, fenv):
    """ Planet mass determined by core mass and atmosphere mass (specified in terms of atm. mass fraction [%]). """
    M_pl = M_core/(1-(fenv/100))
    return M_pl

def calculate_R_env(M_p, fenv, F_p, age, metallicity):
    """ M_p in Earth masses, f_env in percent, Flux in earth units, age in Gyr
    R_env ~ t**0.18 for *enhanced opacities* """
    age_exponent = {"solarZ": -0.11, "enhZ": -0.18}
    R_env = 2.06 * (M_p)**(-0.21) * (fenv/5)**0.59 * (F_p)**0.044 * ((age/1e3)/5)**(age_exponent[metallicity]) # R_earth
    return R_env

def calculate_planet_radius(M_core, fenv, age, flux_solar, metallicity):
    """ description: enhanced opacities; age in Gyr, flux in solar units """
    R_core = calculate_core_radius(M_core)
    M_pl = calculate_planet_mass(M_core, fenv)
    R_env = calculate_R_env(M_pl, fenv, flux_solar, age, metallicity)
    R_pl = R_core + R_env
    return R_pl

def density_planet(M_p, R_p):
    """once I have the radius and a mass estimate for the planet and can estimate a density"""
    rho = (M_p*const.M_earth.cgs/(4./3*np.pi*(R_p*const.R_earth.cgs)**3))
    return rho.value