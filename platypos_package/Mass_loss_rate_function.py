# mass loss rate function

from Lx_evo_and_flux import Lx_evo, flux_at_planet_earth, L_xuv_all, flux_at_planet
import Planet_models_LoFo14 as plmoLoFo14
import Planet_model_Ot20 as plmoOt20
import Beta_K_functions as bk
import astropy.units as u
from astropy import constants as const

def mass_loss_rate_forward_LO14(t_, epsilon, K_on, beta_on, planet_object, f_env, track_dict):
    """Calculate the updated mass-loss rates at any given time step (of the integration) for a 
    Lopez & Fortney (2014) planet with rocky core and H/He envelope.
    Input is a given time t_; this fct. calculates the corresponding Lx(t_) (then Lxuv & Fxuv given a_p), 
    then uses the radius at t_ to estimate a mass (given the current envelope mass fraction fenv);
    then we have a density, can calculate the beta and K parameter (if beta_on and K_on are set to "yes", 
    otherwise at each time step they are beta=1 and/or K=1);
    finally put it all into the mass-loss rate equation to get Mdot at t_
    
    Parameters:
    ----------
    t_:
    Lx_evo:
    calculate_planet_radius:
    epsilon:
    K_on:
    beta_on:
    planet_object:
    f_env: f_env at a given time
    """
    
    Lx = Lx_evo(t=t_, track_dict=track_dict)
    Fxuv = flux_at_planet(L_xuv_all(Lx), planet_object.distance) # get flux at orbital separation a_p
    
    # get planet density
    R_p = plmoLoFo14.calculate_planet_radius(planet_object.core_mass, f_env, t_, 
                                             flux_at_planet_earth(planet_object.Lbol, planet_object.distance),
                                             planet_object.metallicity) 
    M_p = plmoLoFo14.calculate_planet_mass(planet_object.core_mass, f_env) # planetary mass (mass core + mass envelope)
    rho_p = rho = plmoLoFo14.density_planet(M_p, R_p) # initial approx. density

    # specify beta and K
    if beta_on == "yes": # use the beta estimate from Salz et al. 2016
        beta = bk.beta_fct_LO14(M_p, Fxuv, R_p)
    elif beta_on == "no":
        beta = 1.
    if K_on == "yes": # use K-correction estimation from Erkaev et al. 2007
        K = bk.K_fct_LO14(planet_object.distance, M_p, planet_object.mass_star , R_p)
    elif K_on == "no":
        K = 1.
    
    num = 3. * beta**2. * epsilon * Fxuv
    denom = 4. * const.G.cgs * K * rho_p
    M_dot = num / denom
    ################################################################################################################
    ### I define my mass loss rate to be negative, this way the planet loses mass as I integrate forward in time.###
    ### my RK4 function returns the mass evolution of the planet, not the mass lost over time.                   ###
    ################################################################################################################
    return -(M_dot.decompose(bases=u.cgs.bases)).value # mass loss rate in cgs units [g/s]


def mass_loss_rate_forward_Ot20(t_, epsilon, K_on, beta_on, planet_object, radius_at_t, track_dict):
    """Calculate the updated mass-loss rates at any given time step (of the integration) for a 
    planet which follows the "mature" mass-radius relation as given in Otegi et al. (2020).
    Input is a given time t_; this fct. calculates the corresponding Lx(t_) (then Lxuv & Fxuv given a_p), 
    then uses the radius at t_ to estimate a mass based on the volatile-regime mass-radius relation;
    then we have a density, can calculate the beta and K parameter (if beta_on and K_on are set to "yes", 
    otherwise at each time step they are set to beta=1 and/or K=1);
    finally put it all into the mass-loss rate equation to get Mdot at t_
    
    Parameters:
    ----------
    t_:
    Lx_evo:
    calculate_planet_radius:
    epsilon:
    K_on:
    beta_on:
    planet_object:
    f_env: f_env at a given time
    """
    
    Lx = Lx_evo(t=t_, track_dict=track_dict)
    Fxuv = flux_at_planet(L_xuv_all(Lx), planet_object.distance) # get flux at orbital separation a_p
    
    # get planet density
    M_p = plmoOt20.calculate_mass_planet_Ot19(radius_at_t) # estimate planetary mass using a mass-radius relation
    rho_p = rho = plmoOt20.density_planet(M_p, radius_at_t) # initial approx. density
    
    # specify beta and K
    if beta_on == "yes": # use the beta estimate from Salz et al. 2016
        beta = bk.beta_fct_LO14(M_p, Fxuv, radius_at_t)
    elif beta_on == "no":
        beta = 1.
    if K_on == "yes": # use K-correction estimation from Erkaev et al. 2007
        K = bk.K_fct_LO14(planet_object.distance, M_p, planet_object.mass_star , radius_at_t)
    elif K_on == "no":
        K = 1.
    
    num = 3. * beta**2. * epsilon * Fxuv
    denom = 4. * const.G.cgs * K * rho_p
    M_dot = num / denom
    ################################################################################################################
    ### I define my mass loss rate to be negative, this way the planet loses mass as I integrate forward in time.###
    ### my RK4 function returns the mass evolution of the planet, not the mass lost over time.                   ###
    ################################################################################################################
    return -(M_dot.decompose(bases=u.cgs.bases)).value # return the mass loss rate in cgs units [g/s]

