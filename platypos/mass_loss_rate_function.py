import astropy.units as u
from astropy import constants as const

from platypos.lx_evo_and_flux import lx_evo, l_xuv_all
from platypos.lx_evo_and_flux import flux_at_planet_earth, flux_at_planet
import platypos.planet_models_LoFo14 as plmoLoFo14
import platypos.planet_model_Ot20 as plmoOt20
import platypos.beta_K_functions as bk


def mass_loss_rate_forward_LO14(t_, epsilon, K_on, beta_on,
                                planet_object, f_env, track_dict):
    """ Calculate the XUV-induced mass-loss rate at any given time step
    (of the integration) for a Lopez & Fortney (2014) planet with rocky core
    and H/He envelope (see Planet_models_LoFo14 for details).
    (see e.g. Lammer et al. 2003, Owen & Wu 2013; Lopez & Fortney 2013 for
    details on XUV-induced mass loss/ photoevaporation)

    Parameters:
    -----------
    t_ (float): time (in Myr)
    epsilon (float): evaporation efficiency
    K_on (float): set use of K parameter on or off ("on" or "off)
    beta_on (float): set use of beta parameter on or off ("on" or "off)
    planet_object (class obj): object of planet class which contains
                               planetray & stellar parameters (here we read out
                               the core mass, and together with the envelope
                               mass fraction at time t_ we can calcualte the
                               current mass and radius of the planet)
    f_env (float): envelope mass fraction at time t_
    track_dict (dict): dictionary with evolutionary track parameters

    Returns:
    --------
    mass-loss rate in cgs units (g/s) -> NOTE: mass-loss rate is negative!
    """

    # calculate X-ray luminosity and planet flux at time t_
    Lx = lx_evo(t=t_, track_dict=track_dict)
    Fxuv = flux_at_planet(l_xuv_all(Lx), planet_object.distance)

    # calculate current planet density
    R_p = plmoLoFo14.calculate_planet_radius(planet_object.core_mass,
                                             f_env, t_,
                                             flux_at_planet_earth(
                                                    planet_object.Lbol,
                                                    planet_object.distance),
                                             planet_object.metallicity)
    M_p = plmoLoFo14.calculate_planet_mass(planet_object.core_mass, f_env)
    rho_p = rho = plmoLoFo14.density_planet(M_p, R_p)  # mean density

    # specify beta and K
    if beta_on == "yes":
        beta = bk.beta_fct(M_p, Fxuv, R_p)
    elif beta_on == "no":
        beta = 1.
    if K_on == "yes":
        K = bk.K_fct(planet_object.distance, M_p, 
        			 planet_object.mass_star, R_p)
    elif K_on == "no":
        K = 1.

    num = 3. * beta**2. * epsilon * Fxuv
    denom = 4. * const.G.cgs * K * rho_p
    M_dot = num / denom

    # NOTE: I define my mass loss rate to be negative, this way the
    # planet loses mass as I integrate forward in time.
    return -(M_dot.decompose(bases=u.cgs.bases)).value  # in cgs units [g/s]


def mass_loss_rate_forward_Ot20(t_, epsilon, K_on, beta_on,
                                planet_object, radius_at_t, track_dict):
    """ Calculate the XUV-induced mass-loss rate at any given time step
    (of the integration) for a planet which follows the "mature" mass-radius
    relation as given in Otegi et al. (2020) (see Planet_models_Ot20 for
    details).
    (see e.g. Lammer et al. 2003, Owen & Wu 2013; Lopez & Fortney 2013 for
    details on XUV-induced mass loss/ photoevaporation)

    Parameters:
    -----------
    t_ (float): time (in Myr)
    epsilon (float): evaporation efficiency
    K_on (float): set use of K parameter on or off ("on" or "off)
    beta_on (float): set use of beta parameter on or off ("on" or "off)
    planet_object (class obj): object of planet class which contains
                               planetray & stellar parameters
    radius_at_t (float): current planetary radius at time t_; used to estimate
                         the new planet mass at time t_
    track_dict (dict): dictionary with evolutionary track parameters

    Returns:
    --------
    mass-loss rate in cgs units (g/s) -> NOTE: mass-loss rate is negative!
    """

    # calculate X-ray luminosity and planet flux at time t_
    Lx = lx_evo(t=t_, track_dict=track_dict)
    Fxuv = flux_at_planet(l_xuv_all(Lx), planet_object.distance)

    # estimate planetary mass using the Otegi 2020 mass-radius relation
    # and then calculate current mean density
    M_p = plmoOt20.calculate_mass_planet_Ot20(radius_at_t)
    rho_p = rho = plmoOt20.density_planet(M_p, radius_at_t)  # mean density

    # specify beta and K
    if beta_on == "yes":
        beta = bk.beta_fct(M_p, Fxuv, radius_at_t)
    elif beta_on == "no":
        beta = 1.
    if K_on == "yes":
        K = bk.K_fct(planet_object.distance, M_p,
                     planet_object.mass_star, radius_at_t)
    elif K_on == "no":
        K = 1.

    num = 3. * beta**2. * epsilon * Fxuv
    denom = 4. * const.G.cgs * K * rho_p
    M_dot = num / denom

    # NOTE: I define my mass loss rate to be negative, this way the
    # planet loses mass as I integrate forward in time.
    return -(M_dot.decompose(bases=u.cgs.bases)).value  # in cgs units [g/s]
