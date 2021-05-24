import numpy as np
import astropy.units as u
from astropy import constants as const
from collections.abc import Mapping

import platypos.planet_models_LoFo14 as plmoLoFo14
import platypos.planet_models_ChRo16 as plmoChRo16
import platypos.planet_model_Ot20 as plmoOt20
import platypos.beta_K_functions as bk

from platypos.lx_evo_and_flux import lx_evo, l_xuv_all
from platypos.lx_evo_and_flux import flux_at_planet_earth, flux_at_planet


def mass_loss_rate(t_,
                   track_dict,
                   planet,
                   mass_loss_calc="Elim_and_RRlim",
                   epsilon=0.1, K_on="yes",
                   beta_settings={"beta_calc": "off"},
                   f_env=None,
                   radius_at_t_=None,
                   relation_EUV="Linsky"):
    """ 
    Calculate the XUV-induced mass-loss rate at any given time step
    (of the integration) using
    a) a Lopez & Fortney (2014) planet with rocky core and H/He envelope
      (see planet_models_LoFo14 for details) OR
    b) a Chen & Rogers (2016) planet with rocky/icy core and H/He envelope
      (see planet_models_ChRo16 for details) OR
    c) an Otegi et al. (2020) planet, which follows a "mature" mass-radius
      relation"
      (see planet_models_Ot20 for details);
    AND by using one of the the following mass-loss rate calculations:
    1) an energy-limited model
    2) a hydro-based approximation
    3) the radiation-recombination limited formula
    4) a combination of E-lim and RR-lim -> min(E-lim, RR-lim)
    
    See e.g. Lammer et al. 2003, Owen & Wu 2013; Lopez & Fortney 2013 for
    details on XUV-induced mass loss/ photoevaporation;
    OR Kubyshkina et al. 2018, for details on the analytical
    "hydro-based approximation" for the mass-loss rates, which is based on a
    grid of hydro-models (NOTE: Mass-loss rates are calulated using an
    epsilon of 0.15).

    Parameters:
    -----------
    t_ (float): time (in Myr)
    track_dict (dict): dictionary with evolutionary track parameters
                       NOTE: can also be a single Lx value at t_
    planet_object (class obj): object of one of the three planet classes
                               (Planet_LoFo14, Planet_ChRo16, Planet_Ot20),
                               which contains planetray & stellar parameters;
                               e.g. core mass needed to calculate the
                               current mass and radius of the planet)
                               
    mass_loss_calc (str): ADD MORE DETAILS!!
                          "Elim", "RRlim", "Elim_and_RRlim", "HBA"
                          
    epsilon (float): evaporation efficiency
    K_on (float): set use of K parameter on or off ("yes" or "no");
                  default is "yes"
                  
                  
    beta_settings = {"beta_calc": "Salz16",
                     "beta_cut": True,
                     "RL_cut": True}
    
    
    f_env (float): envelope mass fraction at time t_;
                   if not specified: planet is of type Planet_Ot20
    radius_at_t_ (float): if specified: planet is of type Planet_Ot20;
                          otherwise it is automatically set to None and can
                          be ignored
    relation_EUV (str): "Linsky" OR "SanzForcada" -> estimate the EUV luminosity
                        using the X-ray-EUV scaling relations given by
                        Linsky et al. (2013, 2015) OR Sanz-Forcada et al. (2011)

    Returns:
    --------
    mass_loss_Mdot (float): mass-loss rate in cgs units (g/s)
                            -> NOTE: mass-loss rate is negative!
    
    Mdot_info (str): info about what mass-loss rate calculation was used
    """

    # calculate X-ray luminosity and planet flux at time t_
    if isinstance(track_dict, Mapping):
        Lx = lx_evo(t=t_, track_dict=track_dict)
    else:
        Lx = track_dict # a single Lx value at t_ is passed (stored in var track_dict)
    
    Fxuv = flux_at_planet(l_xuv_all(Lx, relation_EUV, planet.mass_star),
                          planet.distance)
    
    #Fx = flux_at_planet(Lx, planet.distance)
    #Feuv = Fxuv - Fx
    #print("{:.3E}, {:.3E}, {:.3E}".format(Fxuv, Fx, Feuv))
    
    # based on the planet model specified, I need to call other functions
    # to calculate the planetary properties (like the radius & density)
    if planet.planet_type == "LoFo14":
        R_p = plmoLoFo14.calculate_planet_radius(
                                        planet.core_mass,
                                        f_env,
                                        t_,
                                        planet.flux,
                                        planet.metallicity)
        M_p = plmoLoFo14.calculate_planet_mass(planet.core_mass, f_env)
        
    elif planet.planet_type == "ChRo16":
        R_p = plmoChRo16.calculate_planet_radius(planet.core_mass,
                                                 f_env,
                                                 planet.flux,
                                                 t_,
                                                 planet.core_comp
                                                 )        
        M_p = plmoChRo16.calculate_planet_mass(planet.core_mass, f_env)
        
    elif planet.planet_type == "Ot20":
        # estimate planetary mass using the Otegi 2020 mass-radius relation
        R_p = radius_at_t_
        M_p = plmoOt20.calculate_mass_planet_Ot20(R_p)
    
    # now, based on the mass_loss_calc specified, I calculate the mass-loss
    # rates using either the energy-limited approximation, the radiation-
    # recombination-limited approximation, a combination of E-lim and RR-lim,
    # or the hydro-based approximation
    if mass_loss_calc == "Elim":
        mass_loss_Mdot = mass_loss_rate_Elim(
                                         planet, 
                                         M_p, R_p, Fxuv,
                                         epsilon, K_on, beta_settings)
        Mdot_info = "Elim"
        
    elif mass_loss_calc == "RRlim":
        mass_loss_Mdot = mass_loss_rate_RRlim(
                                         planet, 
                                         M_p, R_p, Fxuv,
                                         epsilon, K_on, beta_settings)
        Mdot_info = "RRlim"
        
    elif mass_loss_calc == "Elim_and_RRlim":
        mass_loss_Mdot_Elim = mass_loss_rate_Elim(
                                         planet, 
                                         M_p, R_p, Fxuv,
                                         epsilon, K_on, beta_settings)
        mass_loss_Mdot_RRlim = mass_loss_rate_RRlim(
                                         planet, 
                                         M_p, R_p, Fxuv,
                                         epsilon, K_on, beta_settings)
        
        #print("t={:.3f}, E={:.5E}, RR={:.5E}".format(t_, mass_loss_Mdot_Elim, mass_loss_Mdot_RRlim))
        # take the smaller of the two rates (Attention: negative sign!
        if ((np.iscomplex(mass_loss_Mdot_Elim) == True) and \
            (np.iscomplex(mass_loss_Mdot_RRlim) == True)):
                return np.nan
        else:
            mass_loss_Mdot = -min(-mass_loss_Mdot_Elim, -mass_loss_Mdot_RRlim)
            if -mass_loss_Mdot_RRlim < -mass_loss_Mdot_Elim:
                Mdot_info = "RRlim"
            else: 
                Mdot_info = "Elim"
            
    elif mass_loss_calc == "HBA":
        mass_loss_Mdot = mass_loss_rate_HBA(planet, M_p, R_p, Fxuv)
        Mdot_info = "HBA"

    return mass_loss_Mdot, Mdot_info


def mass_loss_rate_Elim(planet, 
                        M_p, R_p, Fxuv,
                        epsilon=0.1, K_on="yes",
                        beta_settings={"beta_calc": "off"}):                                 
    """ Calculates mass-loss rate at a given time using the energy-limited
        approximation. 
    
    beta_settings = {"beta_calc": "off"}
    OR
    beta_settings = {"beta_calc": "Salz16",
                     "beta_cut": True,
                     "RL_cut": True}
        
    """
    M_EARTH = 5.972364730419773e+27 #const.M_earth.cgs.value
    R_EARTH = 637810000.0 #const.R_earth.cgs.value
    G_CONST = 6.674079999999999e-08 #const.G.cgs.value
        
    #rho_p = plmoLoFo14.density_planet(M_p, R_p)
    rho_p = ((M_p * M_EARTH) / (4./3 * np.pi * (R_p * R_EARTH)**3))

    # specify beta
    
    # calculate beta: see beta_calc() for details
    beta_ = bk.beta_calc(M_p, R_p, Fxuv, beta_settings,
                         distance=planet.distance,
                         M_star=planet.mass_star,
                         Lbol_solar=planet.Lbol_solar)
    #print(beta_)
    
    # specify K
    if K_on == "yes":
        K = bk.K_fct(planet.distance, M_p, 
                     planet.mass_star, R_p)
    elif K_on == "no":
        K = 1.

    num = 3. * (beta_)**2. * epsilon * Fxuv
    denom = 4. * G_CONST * K * rho_p
    M_dot = num / denom

    # NOTE: I define my mass loss rate to be negative, this way the
    # planet loses mass as I integrate forward in time.
    return -(M_dot)#.decompose(bases=u.cgs.bases)).value  # in cgs units [g/s]


def mass_loss_rate_HBA(planet, M_p, R_p, Fxuv):
    """ Calculates mass-loss rate at a given time using the hydro-based
        approximation. """

    M_HY = 1.673532836356e-24 #(const.m_p.cgs + const.m_e.cgs).value
    M_EARTH = 5.972364730419773e+27 #const.M_earth.cgs.value
    R_EARTH = 637810000.0 #const.R_earth.cgs.value
    G_CONST = 6.674079999999999e-08 #const.G.cgs.value
    K_B = 1.38064852e-16 #const.k_B.cgs.value
    
    #rho_p = plmoLoFo14.density_planet(M_p, R_p)
    rho_p = ((M_p * M_EARTH) / (4./3 * np.pi * (R_p * R_EARTH)**3))

    # calculate the hydro-based mass-loss rates
    sigma = (15.611 - 0.578 * np.log(Fxuv) \
             + 1.537 * np.log(planet.distance) + 1.018 * np.log(R_p)) \
             / (5.564 + 0.894 * np.log(planet.distance))

    # Jeans escape parameter
    lam = (G_CONST * (M_p * M_EARTH) * M_HY) \
          / (K_B * planet.t_eq * (R_p * R_EARTH))
    
    # choose coefficients of the fit based on e**sigma value
    if np.exp(sigma) > lam:#.decompose(bases=u.cgs.bases).value:
        coeffs = {"beta": 32.0199, "a1": 0.4222, "a2": -1.7489, 
                  "a3": 3.7679, "zeta": -6.8618, "theta": 0.0095}
    elif np.exp(sigma) <= lam:#.decompose(bases=u.cgs.bases).value:
        coeffs = {"beta": 16.4084, "a1": 1.0000, "a2": -3.2861, 
                  "a3": 2.7500, "zeta": -1.2978, "theta": 0.8846}
    try:
        K = coeffs["zeta"] + coeffs["theta"] * np.log(planet.distance)  

        # calculate hydro-based mass loss rate
        M_dot_HBA = np.exp(coeffs["beta"]) * (Fxuv)**coeffs["a1"] \
                    * (planet.distance)**coeffs["a2"] \
                    * (R_p)**coeffs["a3"] \
                    * (lam)**K #.decompose(bases=u.cgs.bases).value)**K

        # NOTE: I define my mass loss rate to be negative, this way the
        # planet loses mass as I integrate forward in time.
        return -M_dot_HBA  # in cgs units [g/s]
    
    except:
        return np.nan
#         raise Exception("Something went wrong in calculating the hydro-based "+\
#                         "mass-loss rates.")  
        
        
def mass_loss_rate_RRlim(planet,
                         M_p, R_p, Fxuv,
                         epsilon=0.1, K_on="yes",
                         beta_settings={"beta_calc": "off"}):
    """ Calculates mass-loss rate at a given time using Radiation-Recombination-
        limited mass-loss rate from Murray-Clay et al. (2009).
        WHY: at high EUV fluxes, F>~10^4 erg/s/cm^2, radiative losses from Ly_alpha
        cooling become important, and mass loss ceases to be Elim.
    .
    .
    .
    """
    M_HY = 1.673532836356e-24#const.m_p.cgs.value + const.m_e.cgs.value
    M_EARTH = 5.972364730419773e+27 #const.M_earth.cgs.value
    R_EARTH = 637810000.0 #const.R_earth.cgs.value
    G_CONST = 6.674079999999999e-08 #const.G.cgs.value
    K_B = 1.38064852e-16 #const.k_B.cgs.value
    EV = 1.6021766208000004e-12 #(1*u.eV).cgs.value
    
    #rho_p = plmoLoFo14.density_planet(M_p, R_p)
    rho_p = ((M_p * M_EARTH) / (4./3 * np.pi * (R_p * R_EARTH)**3))
    
    # calculate beta: see beta() for details
    beta_ = bk.beta_calc(M_p, R_p, Fxuv, beta_settings,
                         distance=planet.distance,
                         M_star=planet.mass_star,
                         Lbol_solar=planet.Lbol_solar)
    
    R_base = R_p * R_EARTH * beta_  # cm
                 
    # 20 eV = 61 nm; typical EUV energy (not integrated over whole spectrum!)
    h_nu0 = 20. * EV
    sigma_nu0 = 6.0*1e-18 * (h_nu0 / (13.6 * EV))**(-3) #cm**2
    T_wind = 1e4 #* u.K # K
    mu_wind = 0.62  # for H/He envelopes
    # isothermal sound speed of fully ionized wind
    c_s = np.sqrt((K_B * T_wind) / (mu_wind * M_HY))
    
    # radius at sonic point unless this gives a value smaller than R_base, then
    # set to R_base
    R_s = ((G_CONST * M_p * M_EARTH) / (2 * c_s**2))
    if R_s < R_base:
        #print('R_s < R_base')
        R_s = R_base
    
    # density at sonic point
    mu_plus_wind = 1.3  # for H/He envelopes
    alpha_rec = 2.70*1e-13 #* (u.cm)**3/u.s  # cm*3/s
    rho_base = mu_plus_wind * M_HY \
                * np.sqrt(((Fxuv * G_CONST * M_p * M_EARTH) \
                   / (h_nu0 * alpha_rec * c_s**2 * R_base**2)))
    rho_s = rho_base * np.exp((G_CONST * M_p * M_EARTH) \
                              / (R_base * c_s**2) * (R_base/R_s - 1.))
    
    M_dot_RR = -4*np.pi * rho_s * c_s * R_s**2
    
    return M_dot_RR#.decompose(bases=u.cgs.bases).value    

# def mass_loss_rate(t_,
#                    track_dict,
#                    planet_object,
#                    mass_loss_calc="Elim",
#                    epsilon=0.1, K_on="yes", beta_on="yes",
#                    f_env=None,
#                    radius_at_t_=None,
#                    beta_cutoff=True,
#                    relation_EUV="Linsky"):
#     """ Calculate the XUV-induced mass-loss rate at any given time step
#     (of the integration) using
#     a) a Lopez & Fortney (2014) planet with rocky core and H/He envelope
#       (see planet_models_LoFo14 for details) OR
#     b) a Chen & Rogers (2016) planet with rocky/icy core and H/He envelope
#       (see planet_models_ChRo16 for details) OR
#     c) an Otegi et al. (2020) planet, which follows a "mature" mass-radius
#       relation"
#       (see planet_models_Ot20 for details);
#     AND by using one of the the following mass-loss rate calculations:
#     1) an energy-limited model
#     2) a hydro-based approximation
#     3) the radiation-recombination limited formula
#     4) a combination of E-lim and RR-lim -> min(E-lim, RR-lim)
    
#     See e.g. Lammer et al. 2003, Owen & Wu 2013; Lopez & Fortney 2013 for
#     details on XUV-induced mass loss/ photoevaporation;
#     OR Kubyshkina et al. 2018, for details on the analytical
#     "hydro-based approximation" for the mass-loss rates, which is based on a
#     grid of hydro-models (NOTE: Mass-loss rates are calulated using an
#     epsilon of 0.15).

#     Parameters:
#     -----------
#     t_ (float): time (in Myr)
#     track_dict (dict): dictionary with evolutionary track parameters
#                        NOTE: can also be a single Lx value at t_
#     planet_object (class obj): object of one of the three planet classes
#                                (Planet_LoFo14, Planet_ChRo16, Planet_Ot20),
#                                which contains planetray & stellar parameters;
#                                e.g. core mass needed to calculate the
#                                current mass and radius of the planet)
                               
#     mass_loss_calc (str): ADD MORE DETAILS!!
#                           "Elim", "RRlim", "Elim_and_RRlim", "HBA"
                          
#     epsilon (float): evaporation efficiency
#     K_on (float): set use of K parameter on or off ("yes" or "no");
#                   default is "yes"
#     beta_on (float): set use of beta parameter on or off ("yes" or "no" or
#                      "yes_old");
#                      ("yes_old" is without Roche Lobe cutoff)
#                      default is "yes"
    
#     f_env (float): envelope mass fraction at time t_;
#                    if not specified: planet is of type Planet_Ot20
#     radius_at_t_ (float): if specified: planet is of type Planet_Ot20;
#                           otherwise it is automatically set to None and can
#                           be ignored
    
#     For planets of type Planet_LoFo14 & Planet_ChRo16 if beta_on == 'yes':
#     - if cutoff == True, beta is kept constant at the lower boundary value
#     for planets with gravities lower than the Salz sample
#     - if cutoff == False, the user agrees to extrapolate this relation beyond
#     the Salz-sample limits.
    
#     relation_EUV (str): "Linsky" OR "SanzForcada" -> estimate the EUV luminosity
#                         using the X-ray-EUV scaling relations given by
#                         Linsky et al. (2013, 2015) OR Sanz-Forcada et al. (2011)

#     Returns:
#     --------
#     mass-loss rate in cgs units (g/s) -> NOTE: mass-loss rate is negative!
#     """

#     # calculate X-ray luminosity and planet flux at time t_
#     if isinstance(track_dict, Mapping):
#         Lx = lx_evo(t=t_, track_dict=track_dict)
#     else:
#         Lx = track_dict # a single Lx value at t_ is passed (stored in var track_dict)
#     Fxuv = flux_at_planet(l_xuv_all(Lx, relation_EUV), planet_object.distance)

#     # based on the planet model specified, I need to call other functions
#     # to calculate the planetary properties (like the radius & density)
#     if planet_object.planet_type == "LoFo14":
#         R_p = plmoLoFo14.calculate_planet_radius(
#                                         planet_object.core_mass,
#                                         f_env,
#                                         t_,
#                                         flux_at_planet_earth(
#                                                 planet_object.Lbol,
#                                                 planet_object.distance),
#                                         planet_object.metallicity)
#         M_p = plmoLoFo14.calculate_planet_mass(planet_object.core_mass, f_env)
        
#     elif planet_object.planet_type == "ChRo16":
#         R_p = plmoChRo16.calculate_planet_radius(planet_object.core_mass,
#                                                  f_env,
#                                                  flux_at_planet_earth(
#                                                         planet_object.Lbol,
#                                                         planet_object.distance),
#                                                  t_,
#                                                  planet_object.core_comp
#                                                  )        
#         M_p = plmoChRo16.calculate_planet_mass(planet_object.core_mass, f_env)
        
#     elif planet_object.planet_type == "Ot20":
#         # estimate planetary mass using the Otegi 2020 mass-radius relation
#         R_p = radius_at_t_
#         M_p = plmoOt20.calculate_mass_planet_Ot20(R_p)
    
    
#     # now, based on the mass_loss_calc specified, I calculate the mass-loss
#     # rates using either the energy-limited approximation, the radiation-
#     # recombination-limited approximation, a combination of E-lim and RR-lim,
#     # or the hydro-based approximation
#     if mass_loss_calc == "Elim":
#         mass_loss_Mdot = mass_loss_rate_Elim(
#                                         planet_object, 
#                                         M_p, R_p, Fxuv,
#                                         epsilon, K_on, beta_on,
#                                         beta_cutoff)
        
#     elif mass_loss_calc == "RRlim":
#         mass_loss_Mdot = mass_loss_rate_RRlim(planet_object, 
#                                               M_p, R_p, Fxuv,
#                                               epsilon, K_on, beta_on,
#                                               beta_cutoff)
        
#     elif mass_loss_calc == "Elim_and_RRlim":
#         mass_loss_Mdot_Elim = mass_loss_rate_Elim(
#                                         planet_object, 
#                                         M_p, R_p, Fxuv,
#                                         epsilon, K_on, beta_on,
#                                         beta_cutoff)
#         mass_loss_Mdot_RRlim = mass_loss_rate_RRlim(
#                                          planet_object, 
#                                          M_p, R_p, Fxuv,
#                                          epsilon, K_on, beta_on,
#                                          beta_cutoff)
#         # take the smaller of the two rates (Attention: negative sign!)
#         if ((np.iscomplex(mass_loss_Mdot_Elim) == True) and \
#             (np.iscomplex(mass_loss_Mdot_RRlim) == True)):
#                 return np.nan
#         else:
#             mass_loss_Mdot = -min(-mass_loss_Mdot_Elim, -mass_loss_Mdot_RRlim)
        
#     elif mass_loss_calc == "HBA":
#         mass_loss_Mdot = mass_loss_rate_HBA(planet_object,
#                                             M_p, R_p, Fxuv)

#     return mass_loss_Mdot



# def mass_loss_rate_Elim(planet_object, 
#                         M_p, R_p, Fxuv,
#                         epsilon=0.1, K_on="yes", beta_on="yes",
#                         beta_cutoff=True):
#     """ Calculates mass-loss rate at a given time using the energy-limited
#         approximation. """
#     M_HY = (const.m_p.cgs + const.m_e.cgs)
#     rho_p = plmoLoFo14.density_planet(M_p, R_p)
    
#     # specify beta
#     if beta_on == "yes":
#         # if XUV radius (as given by the Salz-approximation; or the Salz-cutoff
#         # version) is larger than the planetary Roche lobe radius, we set beta 
#         # such that R_XUV == R_RL
# #         R_RL = planet_object.distance * (const.au/const.R_earth) *\
# #                (M_p / 3 * (M_p + (planet_object.mass_star *\
# #                                   (const.M_sun/const.M_earth))**(1/3)))
#         R_RL = (planet_object.distance*const.au * (M_p*const.M_earth \
#                 / (3.*(M_p*const.M_earth \
#                       + planet_object.mass_star*const.M_sun)))**(1./3)).cgs
#         R_RL = (R_RL / const.R_earth.cgs).value # convert to Earth masses
#         beta = bk.beta_fct(M_p, Fxuv, R_p, beta_cutoff)
#         #print("beta = {:.2f}, R_HE = {:.2f}".format(beta, beta * R_p))
#         if (beta * R_p) > R_RL:
#             #print("R_XUV > R_RL", R_p, (beta * R_p), R_RL)
#             beta = R_RL / R_p
#             #print("beta = {:.2f}, R_HE = {:.2f}\n".format(beta, beta * R_p))
            
#     elif beta_on == "yes_Lo17_old":
#         # second way to estimate R_XUV       
#         surf_grav = (const.G.cgs * (M_p * const.M_earth.cgs)) \
#                     / (R_p *  const.R_earth.cgs)**2
#         mu_below = 2.5  # for H/He envelopes
#         # scaleheight in regime between optical and XUV photosphere
#         H_below = (const.k_B.cgs * planet_object.t_eq *u.K) \
#                     / (mu_below * M_HY * surf_grav)
#         P_photo = 20. * 1e-3*u.bar.cgs  # mbar
#         # following Murray-Clay 2009, estimate the pressure at the tau_XUV=1 boundary
#         # (pressure at base of wind/ XUV photosphere) from the photo-ionization of 
#         # hydrogen
#         h_nu0 = 20*u.eV.cgs  #typical EUV (XUV) energy (not integrated over whole spectrum!)
#         sigma_nu0 = 6.0*1e-18 * (h_nu0 / (13.6*u.eV.cgs))**(-3) * (u.cm)**2 #cm**2
#         P_base = (M_HY * surf_grav) / sigma_nu0
#         R_base = (R_p*const.R_earth.cgs + H_below * \
#                   np.log(P_photo.decompose(bases=u.cgs.bases).value \
#                          / P_base.decompose(bases=u.cgs.bases).value))
        
#         beta = (R_base/const.R_earth.cgs).value / R_p
#         #print("beta = {:.2f}, R_base = {:.2f}".format(beta, (R_base/const.R_earth.cgs).value))
            
#     elif beta_on == "yes_Lo17":
#         # second way to estimate R_XUV (no RL cut)
#         R_RL = (planet_object.distance*const.au * (M_p*const.M_earth \
#                 / (3.*(M_p*const.M_earth \
#                       + planet_object.mass_star*const.M_sun)))**(1./3)).cgs
#         R_RL = (R_RL / const.R_earth.cgs).value # convert to Earth masses
        
#         surf_grav = (const.G.cgs * (M_p * const.M_earth.cgs)) \
#                     / (R_p *  const.R_earth.cgs)**2
#         mu_below = 2.5  # for H/He envelopes
#         # scaleheight in regime between optical and XUV photosphere
#         H_below = (const.k_B.cgs * planet_object.t_eq *u.K) \
#                     / (mu_below * M_HY * surf_grav)
#         P_photo = 20. * 1e-3*u.bar.cgs  # mbar
#         # following Murray-Clay 2009, estimate the pressure at the tau_XUV=1 boundary
#         # (pressure at base of wind/ XUV photosphere) from the photo-ionization of 
#         # hydrogen
#         h_nu0 = 20*u.eV.cgs  #typical EUV (XUV) energy (not integrated over whole spectrum!)
#         sigma_nu0 = 6.0*1e-18 * (h_nu0 / (13.6*u.eV.cgs))**(-3) * (u.cm)**2 #cm**2
#         P_base = (M_HY * surf_grav) / sigma_nu0
#         R_base = (R_p*const.R_earth.cgs + H_below * \
#                   np.log(P_photo.decompose(bases=u.cgs.bases).value \
#                          / P_base.decompose(bases=u.cgs.bases).value))
#         beta = (R_base/const.R_earth.cgs).value / R_p
#         if (beta * R_p) > R_RL:
#             #print("R_XUV > R_RL", R_p, (beta * R_p), R_RL)
#             beta = R_RL / R_p
#         #print("beta = {:.2f}, R_base = {:.2f}".format(beta, beta*R_p))
            
#     elif beta_on == "no":
#         beta = 1.
#         #print("beta = {:.2f}, R_nobeta = {:.2f}".format(beta, beta * R_p))
        
#     elif beta_on == "yes_old":
#         # beta without the Roche-lobe cut-off
#         beta = bk.beta_fct(M_p, Fxuv, R_p, beta_cutoff)
#         #print("beta = {:.2f}, R_Salzbeta = {:.2f}".format(beta, beta * R_p))
    
#     # specify K
#     if K_on == "yes":
#         K = bk.K_fct(planet_object.distance, M_p, 
#                      planet_object.mass_star, R_p)
#     elif K_on == "no":
#         K = 1.

#     num = 3. * beta**2. * epsilon * Fxuv
#     denom = 4. * const.G.cgs * K * rho_p
#     M_dot = num / denom

#     # NOTE: I define my mass loss rate to be negative, this way the
#     # planet loses mass as I integrate forward in time.
#     return -(M_dot.decompose(bases=u.cgs.bases)).value  # in cgs units [g/s]


# def mass_loss_rate_HBA(planet_object, 
#                        M_p, R_p, Fxuv):
#     """ Calculates mass-loss rate at a given time using the hydro-based
#         approximation. """

#     M_HY = (const.m_p.cgs + const.m_e.cgs)
#     rho_p = plmoLoFo14.density_planet(M_p, R_p)
    
#     # calculate the hydro-based mass-loss rates
#     sigma = (15.611 - 0.578 * np.log(Fxuv) \
#              + 1.537 * np.log(planet_object.distance) + 1.018 * np.log(R_p)) \
#              / (5.564 + 0.894 * np.log(planet_object.distance))

#     # Jeans escape parameter
#     lam = (const.G.cgs * (M_p * const.M_earth.cgs) * M_HY) \
#           / (const.k_B.cgs * (planet_object.t_eq * u.K) \
#              * (R_p * const.R_earth.cgs))
    
#     # choose coefficients of the fit based on e**sigma value
#     if np.exp(sigma) > lam.decompose(bases=u.cgs.bases).value:
#         coeffs = {"beta": 32.0199, "a1": 0.4222, "a2": -1.7489, 
#                   "a3": 3.7679, "zeta": -6.8618, "theta": 0.0095}
#     elif np.exp(sigma) <= lam.decompose(bases=u.cgs.bases).value:
#         coeffs = {"beta": 16.4084, "a1": 1.0000, "a2": -3.2861, 
#                   "a3": 2.7500, "zeta": -1.2978, "theta": 0.8846}
#     try:
#         K = coeffs["zeta"] + coeffs["theta"] * np.log(planet_object.distance)  

#         # calculate hydro-based mass loss rate
#         M_dot_HBA = np.exp(coeffs["beta"]) * (Fxuv)**coeffs["a1"] \
#                     * (planet_object.distance)**coeffs["a2"] \
#                     * (R_p)**coeffs["a3"] \
#                     * (lam.decompose(bases=u.cgs.bases).value)**K

#         # NOTE: I define my mass loss rate to be negative, this way the
#         # planet loses mass as I integrate forward in time.
#         return -M_dot_HBA  # in cgs units [g/s]
    
#     except:
#         raise Exception("Something went wrong in calculating the hydro-based "+\
#                         "mass-loss rates.")  
        
        
# def mass_loss_rate_RRlim(planet_object, 
#                          M_p, R_p, Fxuv,
#                          epsilon=0.1, K_on="yes", beta_on="yes",
#                          beta_cutoff=True):
#     """ Calculates mass-loss rate at a given time using Radiation-Recombination-
#         limited mass-loss rate from Murray-Clay et al. (2009).
#         WHY: at high EUV fluxes, F>10^4 erg/s/cm^2, radiative losses from Ly_alpha
#         cooling become important, and mass loss ceases to be Elim.
#     .
#     .
#     .
#     """
#     M_HY = (const.m_p.cgs + const.m_e.cgs)
#     rho_p = plmoLoFo14.density_planet(M_p, R_p)
    
#     # one way to estimate R_XUV (from Salz)
#     # specify beta
#     if beta_on == "yes":
#         # if XUV radius (as given by the Salz-approximation; or the Salz-cutoff
#         # version) is larger than the planetary Roche lobe radius, we set beta 
#         # such that R_XUV == R_RL
#         R_RL = (planet_object.distance*const.au * (M_p*const.M_earth \
#                 / (3.*(M_p*const.M_earth \
#                       + planet_object.mass_star*const.M_sun)))**(1./3)).cgs
#         R_RL = (R_RL / const.R_earth.cgs).value # convert to Earth masses
        
#         beta = bk.beta_fct(M_p, Fxuv, R_p, beta_cutoff)
#         if (beta * R_p) > R_RL:
#             beta = R_RL / R_p
#     elif beta_on == "no":
#         beta = 1.
#     elif beta_on == "yes_old":
#         # beta without the Roche-lobe cut-off
#         beta = bk.beta_fct(M_p, Fxuv, R_p, beta_cutoff)
        
#     # second way to estimate R_XUV (from Murray-Clay/Lopez)
#     surf_grav = (const.G.cgs * (M_p * const.M_earth.cgs)) \
#                 / (R_p *  const.R_earth.cgs)**2
#     mu_below = 2.5  # for H/He envelopes
#     # scaleheight in regime between optical and XUV photosphere
#     H_below = (const.k_B.cgs * planet_object.t_eq *u.K) / (mu_below * M_HY * surf_grav)
#     P_photo = 20. * 1e-3*u.bar.cgs  # mbar
    
#     # following Murray-Clay 2009, estimate the pressure at the tau_XUV=1 boundary
#     # (pressure at base of wind/ XUV photosphere) from the photo-ionization of 
#     # hydrogen
#     h_nu0 = 20*u.eV.cgs  #typical EUV (XUV) energy (not integrated over whole spectrum!)
#     sigma_nu0 = 6.0*1e-18 * (h_nu0 / (13.6*u.eV.cgs))**(-3) * (u.cm)**2 #cm**2
#     P_base = (M_HY * surf_grav) / sigma_nu0
#     R_base = (R_p*const.R_earth.cgs + H_below * \
#               np.log(P_photo.decompose(bases=u.cgs.bases).value \
#                      /P_base.decompose(bases=u.cgs.bases).value))
    
#     #print("R_p = {:.2f}, R_Salzbeta = {:.2f}, R_base = {:.2f}".format(R_p, beta*R_p, (R_base/const.R_earth.cgs).value))  

#     T_wind = 1e4 * u.K # K
#     mu_wind = 0.62  # for H/He envelopes
#     # isothermal sound speed of fully ionized wind
#     c_s = np.sqrt((const.k_B.cgs * T_wind) / (mu_wind * M_HY))
    
#     # radius at sonic point unless this gives a value smaller than R_base, then
#     # set to R_base
#     R_s = ((const.G.cgs * M_p*const.M_earth.cgs) / (2*c_s**2))
#     if R_s < R_base:
#         #print("R_s < R_base: R_s = {:.2f}".format((R_s/const.R_earth.cgs).decompose(bases=u.cgs.bases).value))
#         R_s = R_base
    
#     # density at sonic point
#     mu_plus_wind = 1.3  # for H/He envelopes
#     alpha_rec = 2.70*1e-13 * (u.cm)**3/u.s  # cm*3/s
#     rho_base = mu_plus_wind * M_HY \
#                 * np.sqrt(((Fxuv * const.G.cgs * M_p*const.M_earth.cgs) \
#                    / (h_nu0 * alpha_rec * c_s**2 * R_base**2)))
#     rho_s = rho_base * np.exp((const.G.cgs * M_p*const.M_earth.cgs) \
#                               / (R_base * c_s**2) * (R_base/R_s - 1.))
    
#     M_dot_RR = -4*np.pi * rho_s * c_s * R_s**2
    
#     return M_dot_RR.decompose(bases=u.cgs.bases).value


def mass_loss_rate_noplanetobj(t_, distance, R_p_at_t_, M_p_at_t_, Lx_at_t_, 
                               epsilon=0.1, K_on="yes",
                               beta_settings={"beta_calc": "off"},
                               mass_star=None, Lbol_solar=None,
                               relation_EUV="Linsky",
                               mass_loss_calc="Elim_and_RRlim"):
    """ 
    Pass Lx luminosity directly to calculate mass loss rate at that time.
    
    beta_settings = {"beta_calc": "Salz16",
                     "beta_cut": True,
                     "RL_cut": True}
    
    Calculate the XUV-induced mass-loss rate at any given time step
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

    M_p = M_p_at_t_
    R_p = R_p_at_t_
    Fxuv = flux_at_planet(l_xuv_all(Lx_at_t_, relation_EUV, mass_star), distance)
  
    # now, based on the mass_loss_calc specified, I calculate the mass-loss
    # rates using either the energy-limited approximation, the radiation-
    # recombination-limited approximation, a combination of E-lim and RR-lim,
    # or the hydro-based approximation
    if mass_loss_calc == "Elim":
        mass_loss_Mdot = mass_loss_rate_Elim_noplanetobj(
                                        M_p, R_p, Fxuv, distance,
                                        mass_star, Lbol_solar,
                                        epsilon, K_on, beta_settings)
        
    elif mass_loss_calc == "RRlim":
        mass_loss_Mdot = mass_loss_rate_RRlim_noplanetobj(
                                        M_p, R_p, Fxuv, distance,
                                        mass_star, Lbol_solar,
                                        epsilon, K_on, beta_settings)
        
    elif mass_loss_calc == "Elim_and_RRlim":
        mass_loss_Mdot_Elim = mass_loss_rate_Elim_noplanetobj(
                                        M_p, R_p, Fxuv, distance,
                                        mass_star, Lbol_solar,
                                        epsilon, K_on, beta_settings)
        mass_loss_Mdot_RRlim = mass_loss_rate_RRlim_noplanetobj(
                                        M_p, R_p, Fxuv, distance,
                                        mass_star, Lbol_solar,
                                        epsilon, K_on, beta_settings)
        # take the smaller of the two rates (Attention: negative sign!)
        if ((np.iscomplex(mass_loss_Mdot_Elim) == True) and \
            (np.iscomplex(mass_loss_Mdot_RRlim) == True)):
                return np.nan
        else:
            mass_loss_Mdot = -min(-mass_loss_Mdot_Elim, -mass_loss_Mdot_RRlim)
        
    elif mass_loss_calc == "HBA":
        mass_loss_Mdot = mass_loss_rate_HBA_noplanetobj(M_p, R_p, Fxuv, 
                                                        distance, Lbol_solar)

    return mass_loss_Mdot


def mass_loss_rate_Elim_noplanetobj(M_p, R_p, Fxuv, distance=None,
                                    mass_star=None, Lbol_solar=None,
                                    epsilon=0.1, K_on="yes",
                                    beta_settings={"beta_calc": "off"}):
    """ Calculates mass-loss rate at a given time using the energy-limited
        approximation. 
    
    beta_settings = {"beta_calc": "off"}
    OR
    beta_settings = {"beta_calc": "Salz16",
                     "beta_cut": True,
                     "RL_cut": True}
        
    """
    M_EARTH = 5.972364730419773e+27 #const.M_earth.cgs.value
    R_EARTH = 637810000.0 #const.R_earth.cgs.value
    G_CONST = 6.674079999999999e-08 #const.G.cgs.value
        
    #rho_p = plmoLoFo14.density_planet(M_p, R_p)
    rho_p = ((M_p * M_EARTH) / (4./3 * np.pi * (R_p * R_EARTH)**3))
    
    # specify beta
    # calculate beta: see beta_calc() for details
    beta_ = bk.beta_calc(M_p, R_p, Fxuv, beta_settings,
                    distance=distance,
                    M_star=mass_star,
                    Lbol_solar=Lbol_solar)
    #print(beta_)
    
    # specify K
    if K_on == "yes":
        K = bk.K_fct(distance, M_p, 
                     mass_star, R_p)
    elif K_on == "no":
        K = 1.

    num = 3. * (beta_)**2. * epsilon * Fxuv
    denom = 4. * G_CONST * K * rho_p
    M_dot = num / denom

    # NOTE: I define my mass loss rate to be negative, this way the
    # planet loses mass as I integrate forward in time.
    return -(M_dot)#.decompose(bases=u.cgs.bases)).value  # in cgs units [g/s]


def mass_loss_rate_HBA_noplanetobj(M_p, R_p,
                                   Fxuv,
                                   distance,
                                   Lbol_solar):
    """ Calculates mass-loss rate at a given time using the hydro-based
        approximation. """

    M_HY = 1.673532836356e-24 #(const.m_p.cgs + const.m_e.cgs).value
    M_EARTH = 5.972364730419773e+27 #const.M_earth.cgs.value
    R_EARTH = 637810000.0 #const.R_earth.cgs.value
    G_CONST = 6.674079999999999e-08 #const.G.cgs.value
    K_B = 1.38064852e-16 #const.k_B.cgs.value
    L_SUN = 3.828e+33 #const.L_sun.cgs.value
    SIGMA_SB = 5.6703669999999995e-05 #const.sigma_sb.cgs.value
    AU = 14959787070000.0 #const.au.cgs
    
    #rho_p = plmoLoFo14.density_planet(M_p, R_p)
    rho_p = ((M_p * M_EARTH) / (4./3 * np.pi * (R_p * R_EARTH)**3))
    
    # calculate the hydro-based mass-loss rates
    sigma = (15.611 - 0.578 * np.log(Fxuv) \
             + 1.537 * np.log(distance) + 1.018 * np.log(R_p)) \
             / (5.564 + 0.894 * np.log(distance))

    # Jeans escape parameter
    t_eq = ((Lbol_solar * L_SUN) \
            / (16 * np.pi * SIGMA_SB * (distance * AU)**2))**0.25
    lam = (G_CONST * (M_p * M_EARTH) * M_HY) \
          / (K_B * t_eq * (R_p * R_EARTH))
    
    # choose coefficients of the fit based on e**sigma value
    if np.exp(sigma) > lam:#.decompose(bases=u.cgs.bases).value:
        coeffs = {"beta": 32.0199, "a1": 0.4222, "a2": -1.7489, 
                  "a3": 3.7679, "zeta": -6.8618, "theta": 0.0095}
    elif np.exp(sigma) <= lam:#.decompose(bases=u.cgs.bases).value:
        coeffs = {"beta": 16.4084, "a1": 1.0000, "a2": -3.2861, 
                  "a3": 2.7500, "zeta": -1.2978, "theta": 0.8846}
    try:
        K = coeffs["zeta"] + coeffs["theta"] * np.log(distance)  

        # calculate hydro-based mass loss rate
        M_dot_HBA = np.exp(coeffs["beta"]) * (Fxuv)**coeffs["a1"] \
                    * (distance)**coeffs["a2"] \
                    * (R_p)**coeffs["a3"] \
                    * (lam)**K#.decompose(bases=u.cgs.bases).value)**K

        # NOTE: I define my mass loss rate to be negative, this way the
        # planet loses mass as I integrate forward in time.
        return -M_dot_HBA  # in cgs units [g/s]
    
    except:
        raise Exception("Something went wrong in calculating the hydro-based "+\
                        "mass-loss rates.")  
        
        
def mass_loss_rate_RRlim_noplanetobj(M_p, R_p, Fxuv, distance=None,
                                     mass_star=None, Lbol_solar=None,
                                     epsilon=0.1, K_on="yes",
                                     beta_settings={"beta_calc": "off"}):
    """ Calculates mass-loss rate at a given time using Radiation-Recombination-
        limited mass-loss rate from Murray-Clay et al. (2009).
        WHY: at high EUV fluxes, F>10^4 erg/s/cm^2, radiative losses from Ly_alpha
        cooling become important, and mass loss ceases to be Elim.
    .
    .
    .
    """
    M_HY = 1.673532836356e-24#const.m_p.cgs.value + const.m_e.cgs.value
    M_EARTH = 5.972364730419773e+27 #const.M_earth.cgs.value
    R_EARTH = 637810000.0 #const.R_earth.cgs.value
    G_CONST = 6.674079999999999e-08 #const.G.cgs.value
    K_B = 1.38064852e-16 #const.k_B.cgs.value
    EV = 1.6021766208000004e-12 #(1*u.eV).cgs.value
    
    #rho_p = plmoLoFo14.density_planet(M_p, R_p)
    rho_p = ((M_p * M_EARTH) / (4./3 * np.pi * (R_p * R_EARTH)**3))
    
    # calculate beta: see beta() for details
    beta_ = bk.beta_calc(M_p, R_p,
                         Fxuv, beta_settings,
                         distance=distance,
                         M_star=mass_star,
                         Lbol_solar=Lbol_solar)
    
    R_base = R_p * R_EARTH * beta_  # cm
                  
    h_nu0 = 20. * EV #typical EUV (XUV) energy (not integrated over whole spectrum!)
    sigma_nu0 = 6.0*1e-18 * (h_nu0 / (13.6 * EV))**(-3) #cm**2
    T_wind = 1e4 #* u.K # K
    mu_wind = 0.62  # for H/He envelopes
    # isothermal sound speed of fully ionized wind
    c_s = np.sqrt((K_B * T_wind) / (mu_wind * M_HY))
    
    # radius at sonic point unless this gives a value smaller than R_base, then
    # set to R_base
    R_s = ((G_CONST * M_p * M_EARTH) / (2 * c_s**2))
    if R_s < R_base:
        R_s = R_base
    
    # density at sonic point
    mu_plus_wind = 1.3  # for H/He envelopes
    alpha_rec = 2.70*1e-13 #* (u.cm)**3/u.s  # cm*3/s
    rho_base = mu_plus_wind * M_HY \
                * np.sqrt(((Fxuv * G_CONST * M_p * M_EARTH) \
                   / (h_nu0 * alpha_rec * c_s**2 * R_base**2)))
    rho_s = rho_base * np.exp((G_CONST * M_p * M_EARTH) \
                              / (R_base * c_s**2) * (R_base/R_s - 1.))
    
    M_dot_RR = -4*np.pi * rho_s * c_s * R_s**2
    
    return M_dot_RR#.decompose(bases=u.cgs.bases).value    
        


# def mass_loss_rate_forward_LO14(t_, epsilon, K_on, beta_on,
#                                 planet_object, f_env, track_dict,
#                                 beta_cutoff=False,
#                                 relation_EUV="Linsky"):
#     """ Calculate the XUV-induced mass-loss rate at any given time step
#     (of the integration) for a Lopez & Fortney (2014) planet with rocky core
#     and H/He envelope (see Planet_models_LoFo14 for details).
#     (see e.g. Lammer et al. 2003, Owen & Wu 2013; Lopez & Fortney 2013 for
#     details on XUV-induced mass loss/ photoevaporation)

#     Parameters:
#     -----------
#     t_ (float): time (in Myr)
#     epsilon (float): evaporation efficiency
#     K_on (float): set use of K parameter on or off ("on" or "off)
#     beta_on (float): set use of beta parameter on or off ("on" or "off)
#     planet_object (class obj): object of planet class which contains
#                                planetray & stellar parameters (here we read out
#                                the core mass, and together with the envelope
#                                mass fraction at time t_ we can calcualte the
#                                current mass and radius of the planet)
#     f_env (float): envelope mass fraction at time t_
#     track_dict (dict): dictionary with evolutionary track parameters
#                         NOTE: can also be a single Lx value at t_
                        
#     - if cutoff == True, beta is kept constant at the lower boundary value
#     for planets with gravities lower than the Salz sample
#     - if cutoff == False, the user agrees to extrapolate this relation beyond
#     the Salz-sample limits.

#     Returns:
#     --------
#     mass-loss rate in cgs units (g/s) -> NOTE: mass-loss rate is negative!
#     """

#     # calculate X-ray luminosity and planet flux at time t_
#     if isinstance(track_dict, Mapping):
#         Lx = lx_evo(t=t_, track_dict=track_dict)
#     else:
#         Lx = track_dict # a single Lx value at t_ is passed
#     Fxuv = flux_at_planet(l_xuv_all(Lx, relation_EUV), planet_object.distance)

#     # calculate current planet density
#     R_p = plmoLoFo14.calculate_planet_radius(planet_object.core_mass,
#                                              f_env, t_,
#                                              flux_at_planet_earth(
#                                                     planet_object.Lbol,
#                                                     planet_object.distance),
#                                              planet_object.metallicity)
#     M_p = plmoLoFo14.calculate_planet_mass(planet_object.core_mass, f_env)
#     rho_p = rho = plmoLoFo14.density_planet(M_p, R_p)  # mean density

#     # specify beta and K
#     if beta_on == "yes":
#         # if XUV radius (as given by the Salz-approximation; or the Salz-cutoff
#         # version) is larger than the planetary Roche lobe radius, we set beta 
#         # such that R_XUV == R_RL
#         R_RL = planet_object.distance * (const.au/const.R_earth) * (M_p / 3 * (M_p + \
#                (planet_object.mass_star * (const.M_sun/const.M_earth))**(1/3)))
#         beta = bk.beta_fct(M_p, Fxuv, R_p, beta_cutoff)
#         if (beta * R_p) > R_RL:
#             beta = R_RL / R_p
#     elif beta_on == "no":
#         beta = 1.
#     elif beta_on == "yes_old":
#         # beta without the Roche-lobe cut-off
#         beta = bk.beta_fct(M_p, Fxuv, R_p, beta_cutoff)
    
#     if K_on == "yes":
#         K = bk.K_fct(planet_object.distance, M_p, 
#                      planet_object.mass_star, R_p)
#     elif K_on == "no":
#         K = 1.

#     num = 3. * beta**2. * epsilon * Fxuv
#     denom = 4. * const.G.cgs * K * rho_p
#     M_dot = num / denom

#     # NOTE: I define my mass loss rate to be negative, this way the
#     # planet loses mass as I integrate forward in time.
#     return -(M_dot.decompose(bases=u.cgs.bases)).value  # in cgs units [g/s]


# def mass_loss_rate_forward_LO14_short(t_, epsilon, K_on, beta_on,
#                                       planet_object, f_env, Lx_at_t_,
#                                       beta_cutoff=False,
#                                       relation_EUV="Linsky"):
#     """ 
#     Pass Lx luminosity directly to calculate mass loss rate at that time.
    
#     Calculate the XUV-induced mass-loss rate at any given time step
#     (of the integration) for a Lopez & Fortney (2014) planet with rocky core
#     and H/He envelope (see Planet_models_LoFo14 for details).
#     (see e.g. Lammer et al. 2003, Owen & Wu 2013; Lopez & Fortney 2013 for
#     details on XUV-induced mass loss/ photoevaporation)

#     Parameters:
#     -----------
#     t_ (float): time (in Myr)
#     epsilon (float): evaporation efficiency
#     K_on (float): set use of K parameter on or off ("on" or "off)
#     beta_on (float): set use of beta parameter on or off ("on" or "off)
#     planet_object (class obj): object of planet class which contains
#                                planetray & stellar parameters (here we read out
#                                the core mass, and together with the envelope
#                                mass fraction at time t_ we can calcualte the
#                                current mass and radius of the planet)
#     f_env (float): envelope mass fraction at time t_
#     track_dict (dict): dictionary with evolutionary track parameters

#     Returns:
#     --------
#     mass-loss rate in cgs units (g/s) -> NOTE: mass-loss rate is negative!
#     """

#     Fxuv = flux_at_planet(l_xuv_all(Lx_at_t_, relation_EUV),
#                           planet_object.distance)

#     # calculate current planet density
#     R_p = plmoLoFo14.calculate_planet_radius(planet_object.core_mass,
#                                              f_env, t_,
#                                              flux_at_planet_earth(
#                                                     planet_object.Lbol,
#                                                     planet_object.distance),
#                                              planet_object.metallicity)
#     M_p = plmoLoFo14.calculate_planet_mass(planet_object.core_mass, f_env)
#     rho_p = rho = plmoLoFo14.density_planet(M_p, R_p)  # mean density
        
#     # specify beta and K
#     if beta_on == "yes":
#         # if XUV radius (as given by the Salz-approximation) is larger than
#         # the planetary Roche lobe radius, we set beta such that R_XUV == R_RL
#         R_RL = planet_object.distance * (const.au/const.R_earth) * (M_p / 3 * (M_p + \
#                (planet_object.mass_star * (const.M_sun/const.M_earth))**(1/3)))
#         beta = bk.beta_fct(M_p, Fxuv, R_p, beta_cutoff)
#         if (beta * R_p) > R_RL:
#             beta = R_RL / R_p
#     elif beta_on == "no":
#         beta = 1.
#     elif beta_on == "yes_old":
#         # beta without the Roche-lobe cut-off
#         beta = bk.beta_fct(M_p, Fxuv, R_p, beta_cutoff)
        
#     if K_on == "yes":
#         K = bk.K_fct(planet_object.distance, M_p, 
#         			 planet_object.mass_star, R_p)
#     elif K_on == "no":
#         K = 1.

#     num = 3. * beta**2. * epsilon * Fxuv
#     denom = 4. * const.G.cgs * K * rho_p
#     M_dot = num / denom

#     # NOTE: I define my mass loss rate to be negative, this way the
#     # planet loses mass as I integrate forward in time.
#     return -(M_dot.decompose(bases=u.cgs.bases)).value  # in cgs units [g/s]




# def mass_loss_rate_forward_Ot20(t_, epsilon, K_on, beta_on,
#                                 planet_object, radius_at_t, track_dict):
#     """ Calculate the XUV-induced mass-loss rate at any given time step
#     (of the integration) for a planet which follows the "mature" mass-radius
#     relation as given in Otegi et al. (2020) (see Planet_models_Ot20 for
#     details).
#     (see e.g. Lammer et al. 2003, Owen & Wu 2013; Lopez & Fortney 2013 for
#     details on XUV-induced mass loss/ photoevaporation)

#     Parameters:
#     -----------
#     t_ (float): time (in Myr)
#     epsilon (float): evaporation efficiency
#     K_on (float): set use of K parameter on or off ("on" or "off)
#     beta_on (float): set use of beta parameter on or off ("on" or "off)
#     planet_object (class obj): object of planet class which contains
#                                planetray & stellar parameters
#     radius_at_t (float): current planetary radius at time t_; used to estimate
#                          the new planet mass at time t_
#     track_dict (dict): dictionary with evolutionary track parameters

#     Returns:
#     --------
#     mass-loss rate in cgs units (g/s) -> NOTE: mass-loss rate is negative!
#     """

#     # calculate X-ray luminosity and planet flux at time t_
#     Lx = lx_evo(t=t_, track_dict=track_dict)
#     Fxuv = flux_at_planet(l_xuv_all(Lx), planet_object.distance)

#     # estimate planetary mass using the Otegi 2020 mass-radius relation
#     # and then calculate current mean density
#     M_p = plmoOt20.calculate_mass_planet_Ot20(radius_at_t)
#     rho_p = rho = plmoOt20.density_planet(M_p, radius_at_t)  # mean density

#     # specify beta and K
#     if beta_on == "yes":
#         # if XUV radius (as given by the Salz-approximation) is larger than
#         # the planetary Roche lobe radius, we set beta such that R_XUV == R_RL
#         R_RL = planet_object.distance * (const.au/const.R_earth) * (M_p / 3 * (M_p + \
#                (planet_object.mass_star * (const.M_sun/const.M_earth))**(1/3)))
#         beta = bk.beta_fct(M_p, Fxuv, R_p, beta_cutoff)
#         if (beta * R_p) > R_RL:
#             beta = R_RL / R_p
#     elif beta_on == "no":
#         beta = 1.
#     elif beta_on == "yes_old":
#         # beta without the Roche-lobe cut-off
#         beta = bk.beta_fct(M_p, Fxuv, R_p, beta_cutoff)
    
#     if K_on == "yes":
#         K = bk.K_fct(planet_object.distance, M_p,
#                      planet_object.mass_star, radius_at_t)
#     elif K_on == "no":
#         K = 1.

#     num = 3. * beta**2. * epsilon * Fxuv
#     denom = 4. * const.G.cgs * K * rho_p
#     M_dot = num / denom

#     # NOTE: I define my mass loss rate to be negative, this way the
#     # planet loses mass as I integrate forward in time.
#     return -(M_dot.decompose(bases=u.cgs.bases)).value  # in cgs units [g/s]


# def mass_loss_rate_forward_LO14_HBA(t_, pl, f_env, track_dict, relation_EUV):
#     """ Calculate the XUV-induced mass-loss rate at any given time step
#     (of the integration) for a Lopez & Fortney (2014) planet with rocky core
#     and H/He envelope (see Planet_models_LoFo14 for details).
#     This does not make use of the energy-limited approximation, but instead
#     uses an analytical "hydro-based approximation" presented in 
#     Kubyshkina et al. 2018, which is based on a grid of hydro-models.

#     Parameters:
#     -----------
#     t_ (float): time (in Myr)
#     epsilon (float): evaporation efficiency
#     K_on (float): set use of K parameter on or off ("on" or "off)
#     beta_on (float): set use of beta parameter on or off ("on" or "off)
#     pl (class obj): object of planet class which contains
#                                planetray & stellar parameters (here we read out
#                                the core mass, and together with the envelope
#                                mass fraction at time t_ we can calcualte the
#                                current mass and radius of the planet)
#     f_env (float): envelope mass fraction at time t_
#     track_dict (dict): dictionary with evolutionary track parameters

#     Returns:
#     --------
#     mass-loss rate in cgs units (g/s) -> NOTE: mass-loss rate is negative!
#     """
#     M_HY = (const.m_p.cgs + const.m_e.cgs)

#     # calculate X-ray luminosity and planet flux at time t_
#     if isinstance(track_dict, Mapping):
#         Lx = lx_evo(t=t_, track_dict=track_dict)
#     else:
#         Lx = track_dict # a single Lx value at t_ is passed
#     Fxuv = flux_at_planet(l_xuv_all(Lx, relation_EUV),
#                           pl.distance)

#     # calculate current planet density
#     R_p = plmoLoFo14.calculate_planet_radius(pl.core_mass,
#                                              f_env, t_,
#                                              flux_at_planet_earth(
#                                                         pl.Lbol,
#                                                         pl.distance),
#                                              pl.metallicity)
#     M_p = plmoLoFo14.calculate_planet_mass(pl.core_mass, f_env)
#     rho_p = rho = plmoLoFo14.density_planet(M_p, R_p)  # mean density

#     #print(R_p, f_env)
    
#     sigma = (15.611 - 0.578 * np.log(Fxuv) + 1.537 * np.log(pl.distance) \
#             + 1.018 * np.log(R_p)) / (5.564 + 0.894 * np.log(pl.distance))

#     #print(sigma)
    
#     # Jeans escape parameter
#     lam = (const.G.cgs * (M_p * const.M_earth.cgs) * M_HY) \
#           / (const.k_B.cgs * (pl.t_eq * u.K) * (R_p * const.R_earth.cgs))
    
#     # choose coefficients of the fit based on e**sigma value
#     if np.exp(sigma) > lam.decompose(bases=u.cgs.bases).value:
#         coeffs = {"beta": 32.0199, "a1": 0.4222, "a2": -1.7489, 
#                   "a3": 3.7679, "zeta": -6.8618, "theta": 0.0095}
#     elif np.exp(sigma) <= lam.decompose(bases=u.cgs.bases).value:
#         coeffs = {"beta": 16.4084, "a1": 1.0000, "a2": -3.2861, 
#                   "a3": 2.7500, "zeta": -1.2978, "theta": 0.8846}
#     try:
#         K = coeffs["zeta"] + coeffs["theta"] * np.log(pl.distance)  

#         # calculate hydro-based mass loss rate
#         M_dot_HBA = np.exp(coeffs["beta"]) * (Fxuv)**coeffs["a1"] \
#                     * (pl.distance)**coeffs["a2"] * (R_p)**coeffs["a3"] \
#                     * (lam.decompose(bases=u.cgs.bases).value)**K

#         # NOTE: I define my mass loss rate to be negative, this way the
#         # planet loses mass as I integrate forward in time.
#         return -M_dot_HBA  # in cgs units [g/s]
    
#     except:
#         return np.nan
    
    
# def mass_loss_rate_forward_Ot20_HBA(t_, pl, radius_at_t, track_dict):
#     """ Calculate the XUV-induced mass-loss rate at any given time step
#     (of the integration) for a planet which follows the "mature" mass-radius
#     relation as given in Otegi et al. (2020) (see Planet_models_Ot20 for
#     details).
#     (see e.g. Lammer et al. 2003, Owen & Wu 2013; Lopez & Fortney 2013 for
#     details on XUV-induced mass loss/ photoevaporation)

#     Parameters:
#     -----------
#     t_ (float): time (in Myr)
#     epsilon (float): evaporation efficiency
#     K_on (float): set use of K parameter on or off ("on" or "off)
#     beta_on (float): set use of beta parameter on or off ("on" or "off)
#     pl (class obj): object of planet class which contains
#                                planetray & stellar parameters
#     radius_at_t (float): current planetary radius at time t_; used to estimate
#                          the new planet mass at time t_
#     track_dict (dict): dictionary with evolutionary track parameters

#     Returns:
#     --------
#     mass-loss rate in cgs units (g/s) -> NOTE: mass-loss rate is negative!
#     """
#     M_HY = (const.m_p.cgs + const.m_e.cgs)

#     # calculate X-ray luminosity and planet flux at time t_
#     Lx = lx_evo(t=t_, track_dict=track_dict)
#     Fxuv = flux_at_planet(l_xuv_all(Lx), pl.distance)

#     # estimate planetary mass using the Otegi 2020 mass-radius relation
#     # and then calculate current mean density
#     M_p = plmoOt20.calculate_mass_planet_Ot20(radius_at_t)
#     R_p = radius_at_t
    
#     sigma = (15.611 - 0.578 * np.log(Fxuv) + 1.537 * np.log(pl.distance) \
#             + 1.018 * np.log(R_p)) / (5.564 + 0.894 * np.log(pl.distance))
    
#     # Jeans escape parameter
#     lam = (const.G.cgs * (M_p * const.M_earth.cgs) * M_HY) \
#           / (const.k_B.cgs * (pl.t_eq * u.K) * (R_p * const.R_earth.cgs))
    
#     # choose coefficients of the fit based on e**sigma value
#     if np.exp(sigma) > lam.decompose(bases=u.cgs.bases).value:
#         coeffs = {"beta": 32.0199, "a1": 0.4222, "a2": -1.7489, 
#                   "a3": 3.7679, "zeta": -6.8618, "theta": 0.0095}
#     elif np.exp(sigma) <= lam.decompose(bases=u.cgs.bases).value:
#         coeffs = {"beta": 16.4084, "a1": 1.0000, "a2": -3.2861, 
#                   "a3": 2.7500, "zeta": -1.2978, "theta": 0.8846}
#     try:
#         K = coeffs["zeta"] + coeffs["theta"] * np.log(pl.distance)  

#         # calculate hydro-based mass loss rate
#         M_dot_HBA = np.e**coeffs["beta"] * (Fxuv)**coeffs["a1"] \
#                     * (pl.distance)**coeffs["a2"] * (R_p)**coeffs["a3"] \
#                     * (lam.decompose(bases=u.cgs.bases).value)**K

#         # NOTE: I define my mass loss rate to be negative, this way the
#         # planet loses mass as I integrate forward in time.
#         return -M_dot_HBA  # in cgs units [g/s]
    
#     except:
#         return np.nan