import numpy as np

import astropy.units as u
from astropy import constants as const


def calculate_mass_planet_Ot20(R_pl):
    """ Estimate a planet mass (in Earth masses) based on the mass-radius
    relation derived for the volatile rich regime in Otegi et al. 2020
    (radius planet larger than 2.115 Earth radii, or density < 3.3 g/cc).
    Mass has to be smaller than 120 Earth masses. Radius should be given
    in Earth radii.
    """

    M_earth = const.M_earth.cgs.value
    R_earth = const.R_earth.cgs.value

    if (type(R_pl) == int) or (type(R_pl) == float) \
       or (type(R_pl) == np.float64):  # if R is single value
        M_p_volatile = 1.74*(R_pl)**1.58  # if rho < 3.3 g/cm^3
        rho_volatile = (M_p_volatile*M_earth) \
                        / (4/3*np.pi*(R_pl*R_earth)**3)
        if (rho_volatile >= 3.3):
            raise Exception("Planet with this radius is too small and"\
				 			+ " likely rocky; use LoFo14 models instead.")
        else:
            if (M_p_volatile >= 120):
                raise Exception("Planet too massive. M-R relation only valid"\
                                + " for <120 M_earth.")
            else:
                return M_p_volatile

    elif len(R_pl) > 1:  # if R is array
        Ms = []
        for i in range(len(R_pl)):
            M_p_volatile_i = 1.74*(R_pl[i])**1.58  # if rho < 3.3 g/cm^3
            rho_volatile_i = (M_p_volatile_i * M_earth) \
                             / (4/3*np.pi*(R_pl[i]*R_earth)** 3)
            if (rho_volatile_i >= 3.3) or (M_p_volatile_i >= 120):
                M_i = np.nan
            else:
                M_i = M_p_volatile_i
            Ms.append(M_i)
        Ms = np.array(Ms)
        return Ms


def calculate_radius_planet_Ot20(M_pl):
    """ Estimate a planet radius (in Earth radii) based on the mass-radius
    relation derived for the volatile rich regime in Otegi et al. 2020.
    Mass needs to be between ~5.7 and 120 Earth masses and should be given 
    in Earth masses.
    """

    M_earth = const.M_earth.cgs.value
    R_earth = const.R_earth.cgs.value

    if (type(M_pl) == int) or (type(M_pl) == float) \
            or (type(M_pl) == np.float64):  # if M is single value
        # R_p_volatile = 0.7*M_pl**0.63 # if rho < 3.3 g/cm^3
        R_p_volatile = (M_pl/1.74)**(1./1.58)
        rho_volatile = (M_pl*M_earth) \
                        / (4/3*np.pi*(R_p_volatile*R_earth)**3)
        if (rho_volatile >= 3.3):
            raise Exception("Planet with this mass/radius is too small and"\
                            + " likely rocky; use LoFo14 models instead.")
        else:
            if (M_pl >= 120):
                raise Exception("Planet too massive. M-R relation only"\
                                + " valid for <120 M_earth.")
            else:
                return R_p_volatile

    elif len(M_pl) > 1:  # if M is array
        Rs = []
        for i in range(len(M_pl)):
            # R_p_volatile_i = 0.7*M_pl[i]**0.63 # if rho < 3.3 g/cm^3
            R_p_volatile_i = (M_pl[i]/1.74)**(1./1.58)
            rho_volatile_i = (M_pl[i]*M_earth)\
                            / (4/3*np.pi*(R_p_volatile_i*R_earth)**3)
            if (rho_volatile_i >= 3.3) or (M_pl[i] >= 120):
                R_i = np.nan
            else:
                R_i = R_p_volatile_i
            Rs.append(R_i)
        Rs = np.array(Rs)
        return Rs


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
