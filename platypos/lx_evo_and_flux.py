import numpy as np
import astropy.units as u
from astropy import constants as const
import scipy.optimize as optimize
from scipy.optimize import fsolve


def lx_evo(t, track_dict):
    """
    Function to calculate the X-ray luminosity, Lx, of the host star 
    at a given time t.
    NOTE: this function is not very pretty and could use some
    	  cosmetic updates.
    
    NOTE: I tried to set up this function such that it works going 
    forward (for young system) and backwards (for older system) in time;
    -> so t_curr(ent) is a little confusing! -> we have two cases!!
    
    Parameters Case 1 - a young system which you want to evolve forward in time 
    (i.e. estimate the mass loss the planet will undergo in the future):
    ---------------------------------------------------------------------------
    t_start (float): Starting time for the forward simulation in Myr 
    				 (e.g. 23 Myr for V1298 Tau)
    
    Lx_max (float): Measured Lx value at the system age (i.e. t_start)
    
    t_curr (float): Set to 1 Gyr (with Lx_curr = Lx at 1 Gyr), which is 
    				where, by design, all activity tracks converge
    				(based on Tu et al. (2015)/ Johnstone et al.(2015) models);

    Lx_curr (float): Lx value at 1 Gyr taken from Tu et al. (2015)
    
    t_5Gyr (float): Defines, together with Lx_5Gyr, the slope common for all 
    				activity evolution tracks past 1 Gyr 
    				(also based on Tu et al.(2015));
    
    Lx_5Gyr (float): Defines, together with t_5Gyr, the slope common for all 
    				 activity evolution tracks past 1 Gyr 
    				 (also based on Tu et al.(2015));
    Note: in principle these two pairs (t_curr, Lx_1Gyr) and (t_5Gyr, Lx_5Gyr)
    can be set however the user wants.
    
    # Case 2 not implemented yet in Platypos
    Parameters Case 2 - an older system, which you want to evolve backwards in time 
    (i.e. estimate the mass loss the planet has undergone until now):
    ---------------------------------------------------------------------------
    t_start is the youngest age you want to calculate backwards 
    (i.e. this shouldbe sth. like ~10 Myr, or close to the disk clearing 
    timescale); 

    Lx_max (float): Some saturation X-ray luminosity 
    				(e.g. Lx,sat/Lbol ~ 10^(-3))

    t_curr (float): Needs to be set to the system age 
    				(e.g. t_curr=700 Myr for K2-136b); 

    Lx_curr (float): If a measured Lx value is available for the system, 
    				 otherwise need to estimate;
    t_5Gyr (float): Defines, together with Lx_5Gyr, the slope common for all
        			activity evolution tracks past 1 Gyr 
        			(also based on Tu et al.(2015));
    Lx_5Gyr (float): Defines, together with t_5Gyr, the slope common for all
        			 activity evolution tracks past 1 Gyr 
        			 (also based on Tu et al.(2015));

    
    Returns:
    --------
    Lx_at_time_t (float) Returns the corresponding Lx value at given time t
    """
    
    # read in the parameters from the provided dictionary
    # -> these are all the parameters required to define the tracks
    t_start, t_sat=  track_dict["t_start"], track_dict["t_sat"]
    t_curr, t_5Gyr = track_dict["t_curr"], track_dict["t_5Gyr"]
    Lx_max = track_dict["Lx_max"]
    Lx_curr, Lx_5Gyr = track_dict["Lx_curr"], track_dict["Lx_5Gyr"]
    dt_drop = track_dict["dt_drop"]
    Lx_drop_factor = track_dict["Lx_drop_factor"]
    
    # Define function for calculating a power law
    powerlaw = lambda x, k, index: k * (x**index)
    
    # t_curr has to be greater than t_start
    if (t_curr < t_start) or (t < t_start):
        if t <= t_start: # produce a saturation regime before t_start
            Lx = Lx_max
        else:
            print("Make sure t_curr > t_start, t >= t_start")
    
    elif t > t_curr:
        # this is the regime past 1 Gyr (the same for all Tu tracks)
        # I use the Tu 2015 slope which is approx the same for all 
        # three evolutionary tracks (and based on current solar value);
        # this is mainly for consistency since we approximate our tracks
        # based on the Tu-tracks
        
        # exception for the OwWu17 case, where the slope past t_sat is c
        # onstant for all t > t_sat
        if t_5Gyr == t_curr: # then we are dealing with the OwWu17 case
            alpha = (np.log10(Lx_curr/Lx_max))/(np.log10(t_curr/t_sat))
            k = 10**(np.log10(Lx_max) - alpha*np.log10(t_sat))
            Lx = powerlaw(t, k, alpha)

        else: # normal case, i.e. slope past t_curr given by input parameters
            alpha = (np.log10(Lx_5Gyr/Lx_curr))/(np.log10(t_5Gyr/t_curr))
            k = 10**(np.log10(Lx_curr) - alpha*np.log10(t_curr))
            Lx = powerlaw(t, k, alpha)
        
        return Lx
    
    elif t_curr > t_start:
        # dt_drop==0 means we create a track with only the saturation regime
        # and a single power-law-slope drop to the converging Lx at t_curr
        if dt_drop==0: # then t_sat == t_drop
            if t_start >= t_sat: # then t_sat <= t_start < t_curr
                if t >= t_sat: #and t <= t_curr:
                    # only data in power law regime
                    alpha = (np.log10(Lx_curr / Lx_max)) \
                    		 / ( np.log10(t_curr / t_sat))
                    k = 10**(np.log10(Lx_max) - alpha * np.log10(t_sat))
                    Lx = powerlaw(t, k, alpha)
                    
            elif t_start < t_sat: # then t_start < t_sat < t_curr
                if t > t_sat: #and t <= t_curr:
                    alpha = (np.log10(Lx_curr / Lx_max)) \
                    		 / (np.log10(t_curr / t_sat))
                    k = 10**(np.log10(Lx_max) - alpha * np.log10(t_sat))
                    Lx = powerlaw(t, k, alpha)
                elif t <= t_sat:
                    Lx = Lx_max
                    
        elif dt_drop > 0: # then t_sat != t_drop
            # dt_drop > 0 means we create a more fancy track with the 
            # saturation regime and then a drop with slope change to 
            # the converging Lx at 1 Gyr (Lx track with three regimes)
            t_drop = t_sat + dt_drop # t_sat < t_drop
            
            if t <= t_sat:
                Lx = Lx_max
            elif (t > t_sat) and (t <= t_drop): # first of the two slopes
                alpha_drop1 = (np.log10((Lx_max / Lx_drop_factor) / Lx_max)) \
                			  / (np.log10(t_drop / t_sat))
                k_drop1 = 10**(np.log10(Lx_max) - alpha_drop1 \
                			   * np.log10(t_sat))
                Lx = powerlaw(t, k_drop1, alpha_drop1)
                
            elif t > t_drop:
                alpha_drop2 = (np.log10(Lx_curr / (Lx_max / Lx_drop_factor))) \
                			   / (np.log10(t_curr / t_drop))
                k_drop2 = 10**(np.log10((Lx_max / Lx_drop_factor)) \
                			   - alpha_drop2 * np.log10(t_drop))
                Lx = powerlaw(t, k_drop2, alpha_drop2)     
    return Lx

def flux_at_planet_earth(L, a_p):
    """Function calculates the flux that the planet recieves at a given 
    distance normalized with Earth's flux -> the semi-major axis is used
    for the distance (the eccentricity of the planetary orbits is ignored)
    """
    FLUX_AT_EARTH = 1373. * 1e7 * (u.erg/u.s) / (100*u.cm*100*u.cm) # erg/s/cm^2

    A = (4. * np.pi * (a_p * const.au.cgs)**2)
    F = (L * u.erg / u.s) / A
    F_earth = (F/FLUX_AT_EARTH).value
    return F_earth

def flux_at_planet(L, a_p):
    """Function calculates the flux that the planet recieves at a given 
    distance (in erg/s/cm^2) -> the semi-major axis is used for the distance
    (the eccentricity of the planetary orbits is ignored)
    """
    A = (4. * np.pi * (a_p * const.au.cgs)**2)
    F = ((L * u.erg / u.s) / A).value
    return F # erg/s/cm^2

def l_xuv_all(Lx):
    """Function to estimate the EUV luminosity using the scaling relations 
    given by Sanz-Forcada et al. (2011); the measured X-ray luminosity can 
    be extrapolated to the total high-energy radiation as given below. 
    To get the combined high-energy luminosity (L_xuv), we add L_x and L_euv.
    THEY USE: Lx(0.1-2.4 keV) ROSAT band
    """
    log_L_euv = 0.860 * np.log10(Lx) + 4.80
    Leuv = 10.**log_L_euv
    log_L_xuv = np.log10(Lx + Leuv)
    return 10.**log_L_xuv  # erg/s

def undo_what_Lxuv_all_does(L_xuv):
    """ If you have L_XUV given, this takes the Sanz-Forcada et al. (2011) 
    scaling relation and reverses it to estimate Lx.
    Function needs better implementation!
    """
    def Calculate_Lx(Lx):
        return Lx + 10**(0.86 * np.log10(Lx) + 4.8) - L_xuv

    if (L_xuv > 1. * 10**29):
        f_guess = L_xuv
    elif (L_xuv <= 1. * 10**29) and (L_xuv > 1. * 10**26):
        f_guess = 1. * 10**25
    elif (L_xuv <= 1. * 10**26):
        f_guess = 1. * 10**22

    Lx = optimize.fsolve(Calculate_Lx, x0=f_guess)[0]
    return Lx

#def Lx_relation_Booth(t, R_star):
#    """Gives Lx slope beyond 1 Gyr based on measurements
#		(see Booth et al. 2017). R_star in terms of solar units
#	 """
#    log_Lx = 54.65 - 2.8 * np.log10(t) + 2 * np.log10(R_star)
#    return 10.**log_Lx

def calculate_Lx_sat(L_star_bol):
    """ Typical relation (from observations) to estimate the 
    saturation X-ray luminosity given a bolometric luminosity.
    """
    return 10**(-3) * (L_star_bol * const.L_sun.cgs).value


def l_high_energy(t_i, mass_star):
    """ L_HE (UV to X-ray) as in Owen & Wu (2017).
    Parameters:
    -----------
    t_i (float): time in Myr
    mass_star (float): stellar mass in solar masses
    Returns:
    --------
    L_HE (float): X+UV luminosity in erg/s
    """
    t_sat = 100. # Myr
    L_sat = 10**(-3.5)*(mass_star) * (const.L_sun.cgs).value
    a_0 = 0.5
    if t_i < t_sat:
        L_HE = L_sat
    elif t_i >= t_sat:
        L_HE = L_sat*(t_i/t_sat)**(-1-a_0)
    return L_HE # in erg/s