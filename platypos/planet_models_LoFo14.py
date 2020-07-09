# Lopez & Fortney 2014 models
import numpy as np
import warnings
import astropy.units as u
from astropy import constants as const

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# ignoring this warining is certainly not the best way, but it is currently
# thrown when one of the "in-between" time steps inside the Runge-Kutta
# integration method results in a planet which has no atmosphere left.


def calculate_core_radius(M_core):
    """ M-R relation for rock/iron Earth-like core. (no envelope)
    (see Lopez & Fortney 2014 for details.)
    """
    R_core = (M_core**0.25)
    return R_core


def calculate_planet_mass(M_core, fenv):
    """ Planet mass determined by core mass and atmosphere mass
    (specified in terms of envelope mass fraction [in % !]). """
    M_pl = M_core/(1-(fenv/100))
    return M_pl


def calculate_R_env(M_p, fenv, F_p, age, metallicity):
    """ Calculate planetary envelope radius using the parametrized
    results from Lopez & Fortney 2014 for planets with rocky core
    and H/He gaseous envelope.

    Parameters:
    -----------
    M_p (float): planet mass (in Earth masses)
    f_env (float): envelope mass fraction (given in percent!)
    F_p (float): bolometric flux revcieved by the planet (in units
                 of Earth's insolation)
    age (float): age of the planet (in Myr)
    metallicity (str): chose models with solar or enhanced metallicity
                       (set to "solarZ" or "enhZ")
    (check Lopez & Fortney 2014 for further details and range of model
    validity).

    Returns:
    --------
    R_env (float): radius of the envelope (in Earth radii)
    """
    age_exponent = {"solarZ": -0.11, "enhZ": -0.18}
    R_env = 2.06 * (M_p)**(-0.21) * (fenv/5)**0.59 * (F_p)**0.044 \
        * ((age/1e3)/5)**(age_exponent[metallicity])
    return R_env  # R_earth


def calculate_planet_radius(M_core, fenv, age,
                            flux_solar, metallicity):
    """ Calculate planetary radius (core + envelope) using the
    parametrized results from Lopez & Fortney 2014 for planets with
    rocky core and H/He gaseous envelope.

    Parameters:
    -----------
    M_core (float): core mass (in Earth masses)
    f_env (float): envelope mass fraction (given in percent!)
    flux_solar (float): bolometric flux revcieved by the planet
                        (in units of Earth's flux)
    age (float): age of the planet (in Myr)
    metallicity (str): chose models with solar or enhanced metallicity
                       (set to "solarZ" or "enhZ")
    (check Lopez & Fortney 2014 for further details and range of model
    validity).

    Returns:
    --------
    R_pl (float): radius of the planet (in Earth radii)
    """

    R_core = calculate_core_radius(M_core)
    M_pl = calculate_planet_mass(M_core, fenv)
    R_env = calculate_R_env(M_pl, fenv, flux_solar, age, metallicity)
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

    rho = (M_p*const.M_earth.cgs) \
        / (4./3*np.pi*(R_p*const.R_earth.cgs)**3)
    return rho.value
