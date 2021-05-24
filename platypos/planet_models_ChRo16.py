# Chen & Rogers 2016 models
import numpy as np
import astropy.units as u
from astropy import constants as const


def calculate_planet_mass(M_core, fenv):
    """ Planet mass determined by core mass and atmosphere mass
    (specified in terms of atm. mass fraction (M_env/M_pl) [%]). """
    M_pl = M_core / (1 - (fenv/100))
    return M_pl


def calculate_core_radius(M_core, rock_or_ice="rock"):
    """ M-R relation for rock/iron Earth-like or icy core (no envelope).
    See Chen & Rogers (2016) for details. 
    Parameters:
    M_core (float): core mass in Earth masses
    rock_or_ice (str): 'rock' (default) or 'ice'
    """
    if rock_or_ice == "rock":
        return 0.97 * M_core**0.28 # in R_earth
    elif rock_or_ice == "ice":
        return 1.27 * M_core**0.27 # in R_earth


def calculate_R_env(M_core, fenv, F_p, age, rock_or_ice="rock"):
    """ Calculate planetary envelope radius using the parametrized
    results from Chen & Rogers (2016) for planets with rocky/icy core
    and H/He gaseous envelope.

    Parameters:
    -----------
    M_core (float):  core mass in Earth masses
    fenv (float): envelope mass fraction (given in percent!)
    F_p (float): bolometric flux revcieved by the planet (in units
                 of Earth's insolation)
    age (float): age of the planet (in Myr)
    rock_or_ice (str): 'rock' (default) or 'ice'
    
    (check Chen & Rogers (2016) for further details and range of model
    validity).

    Returns:
    --------
    R_env (float): radius of the envelope (in Earth radii)"""
    
    # Coefficients Rocky-Core Planets
    dict_rock = {"c_0": 0.131, "c_1": -0.348, "c_2": 0.631, "c_3": 0.104,
                 "c_4": -0.179, "c_12": 0.028, "c_13": -0.168, "c_14": 0.008,
                 "c_23": -0.045, "c_24": -0.036, "c_34": 0.031, "c_11": 0.209,
                 "c_22": 0.086, "c_33": 0.052, "c_44": -0.009}
    # Coefficients Ice-Rock-Core Planets
    dict_ice = {"c_0": 0.169, "c_1": -0.436, "c_2": 0.572, "c_3": 0.154,
                "c_4": -0.173, "c_12": 0.014, "c_13": -0.210, "c_14": 0.006,
                "c_23": -0.048, "c_24": -0.040, "c_34": 0.031, "c_11": 0.246,
                "c_22": 0.074, "c_33": 0.059, "c_44": -0.006}
    
    if rock_or_ice == "rock":
        params = dict_rock
    elif rock_or_ice == "ice":
        params = dict_ice
    
    R_core = calculate_core_radius(M_core, rock_or_ice)
    M_pl = calculate_planet_mass(M_core, fenv)
    
    x1 = np.log10(M_pl)
    x2 = np.log10((fenv/100) / 0.05)
    x3 = np.log10(F_p)
    x4 = np.log10((age/1e3) / 5)  # convert age to Gyr
    var = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}
    
    def sum_notation1(i0=1, end=4):
        total_sum = 0.
        for i in range(i0, end+1):
            c_i = params["c_"+str(i)]
            x_i = var["x"+str(i)]
            f = lambda x : c_i*x_i
            total_sum += f(i)
        return total_sum

    def sum_notation2(i0=1, end_i=4, j0=1, end_j=4):
        total_sum = 0.
        for i in range(i0, end_i+1):
            for j in range(j0, end_j+1):
                try: 
                    c_ij = params["c_"+str(i)+str(j)]
                    x_i = var["x"+str(i)]
                    x_j = var["x"+str(j)]
                    f = lambda x : c_ij*x_i*x_j
                    total_sum += f(i)
                except KeyError:
                    #print("Sth. went wrong if you see this!")
                    continue
        return total_sum
    
    # calculate envelope radius using the two sum functions
    log10_Renv = params["c_0"] + sum_notation1(1, 4) + sum_notation2(1, 4, 1, 4)
    R_env = 10**log10_Renv # in Earth radii
    return R_env


def calculate_planet_radius(M_core, fenv, F_p, age, rock_or_ice="rock"):
    """ Calculate planetary radius (core + envelope) using the
    parametrized results from Chen & Rogers 2016 for planets with
    rocky/icy core and H/He gaseous envelope.

    Parameters:
    -----------
    M_core (float): core mass (in Earth masses)
    f_env (float): envelope mass fraction (given in percent!)
    F_p (float): bolometric flux revcieved by the planet
                        (in units of Earth's flux)
    age (float): age of the planet (in Myr)
    rock_or_ice (str): 'rock' (default) or 'ice'
    
    (check Chen & Rogers 2016 for further details and range of model
    validity).

    Returns:
    --------
    R_pl (float): radius of the planet (in Earth radii)
    """
    
    R_core = calculate_core_radius(M_core, rock_or_ice)
    M_pl = calculate_planet_mass(M_core, fenv)
    R_env = calculate_R_env(M_core, fenv, F_p, age, rock_or_ice)
    R_pl = R_core + R_env
    return R_pl


def density_planet(M_p, R_p):
    """ Calculate the mean density in cgs units using the radius and
    mass of the planet.

    Parameters:
    -----------
    M_p (float): mass of the planet (in Earth masses)
    R_p (float): radius of the planet (in Earth radii)

    Returns:
    --------
    rho (float): density in cgs units (g/ccm)
    """
    return (M_p * const.M_earth.cgs.value) \
            / ((4./3) * np.pi * (R_p * const.R_earth.cgs.value)**3)
