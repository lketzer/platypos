import numpy as np
import astropy.units as u
from astropy import constants as const
import scipy.optimize as optimize
from scipy.optimize import fsolve
from scipy import interpolate
from platypos.mass_luminosity_relation import mass_lum_relation_mamajek
from platypos.mass_luminosity_relation import mass_lum_relation_thomas

mass_luminosity_relation = mass_lum_relation_mamajek()
ZAMS_mass_luminosity_relation = mass_lum_relation_thomas()

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
    L in erg/s a in a.u.!!
    """
    FLUX_AT_EARTH = 1373. * 1e7 / (100*100)#(u.erg/u.s) / (100*u.cm*100*u.cm) # erg/s/cm^2

    A = (4. * np.pi * (a_p * const.au.cgs.value)**2)
    return (L / A) / FLUX_AT_EARTH

def flux_at_planet(L, a_p):
    """Function calculates the flux that the planet recieves at a given 
    distance (in erg/s/cm^2) -> the semi-major axis is used for the distance
    (the eccentricity of the planetary orbits is ignored)
    
    L in erg/s a in a.u.!! 
    """
    A = (4. * np.pi * (a_p * const.au.cgs.value)**2)
    return L / A # erg/s/cm^2

# def l_xuv_all(lx):
#     """Function to estimate the EUV luminosity using the scaling relations 
#     given by Sanz-Forcada et al. (2011); the measured X-ray luminosity can 
#     be extrapolated to the total high-energy radiation as given below. 
#     To get the combined high-energy luminosity (L_xuv), we add L_x and L_euv.
#     THEY USE: Lx(0.1-2.4 keV) ROSAT band
#     """
#     log_L_euv = 0.860 * np.log10(lx) + 4.80
#     l_EUV = 10.**log_L_euv
#     log_L_xuv = np.log10(lx + l_EUV)
#     return 10.**log_L_xuv  # erg/s

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


def lx_relation_Booth(t_, mass_star):
    """Gives Lx slope beyond 1 Gyr based on measurements
        (see Booth et al. 2017). R_star in terms of solar units, t_ in Myrs
    """
    R_star = 0.438 * mass_star**2 + 0.479 * mass_star + 0.075 # Eker et al. 2018
    log_Lx = 54.65 - 2.8 * np.log10(t_*1e6) + 2 * np.log10(R_star)
    return 10.**log_Lx

# def calculate_Lx_sat(mass_star, Lx_calculation="Tu15"):
#     """ Estimate the saturation X-ray luminosity given a stellar mass
#     (mass_star in solar masses).
#     Lx_calculation:
#         - "OwWu17"
#         - "OwWu17_X=HE"
#         - "Tu15"
#         - "1e-3"
#     """
#     Lbol = 10**mass_luminosity_relation(mass_star)
    
#     if Lx_calculation == "OwWu17":
#         # OwWu 17
#         # NOTE: since platypos expects a X-ray saturation luminosity,
#         # and not an XUV sat. lum, for the OwWu17 formula we need to
#         # invert Lxuv_all, so that we get the corresponding Lx for all Lxuv
#         # then platpos needs to be run with relationEUV="SanzForcada"!!!
#         L_sat = 10**(-3.5) * mass_star * const.L_sun.cgs.value
#         Lx_sat = undo_what_Lxuv_all_does(L_sat)

#     elif Lx_calculation == "OwWu17_X=HE":
#         # OwWu 17, with Lx,sat = L_HE,sat
#         Lx_sat = 10**(-3.5) * mass_star * const.L_sun.cgs.value

#     elif Lx_calculation == "Tu15":
#         # need Mass-Luminosity relation to estimate L_bol based on the
#         # stellar mass (NOTE: we use a MS M-R raltion and ignore any
#         # PRE-MS evolution
#         Lx_sat = l_high_energy(1.0, mass_star, paper="Tu15")

#     elif Lx_calculation == "1e-3":
#         # need Mass-Luminosity relation to estimate L_bol based on the
#         # stellar mass (NOTE: we use a MS M-R raltion and ignore any
#         # PRE-MS evolution
#         Lbol = 10**mass_luminosity_relation(mass_star)
#         Lx_sat = 10**(-3.0) * Lbol * const.L_sun.cgs.value
                                              
#     elif Lx_calculation == "Kuby20":
#         t_sat = 62.
#         Lx_sat = 886. * Lbol * (4. * np.pi * (const.au.cgs.value)**2) 
#         l_high_energy(t_i, mass_star, paper="OwWu17")
        
    
#     elif
    
    
    
#     return Lx_sat
    
        
def l_high_energy(t_, mass_star=1.0, paper="Tu15",
                  EUV_relation="Linsky", ML_rel="ZAMS_Thomas"):
    """ L_HE (XUV or X-only) at any given time t_. 
    To estimate the saturation X-ray luminosity for a given a stellar mass 
    (in solar masses), input t_=1.0 Myr.
    
    Check parameters for options on Lx or Lxuv calculations!
        
    Sidenote:
    M-L relation:
    - "MS_Mamajeck" - can either be the main-sequence one from Mamajeck
    - "ZAMS_Thomas": or estimate Lbol right when star reaches the ZAMS
                     (from T. Steindl's pre-MS MESA models)
          
    EUV_relation: user has 4 options to estimate the EUV content based on X-ray

        
    Parameters:
    -----------
    t_i (float): time in Myr
    mass_star (float): stellar mass in solar masses
    paper (str): "OwWu17XUV", "WaDa18XUV", "Kuby20XUV", "LoRi17XUV",
                 "Tu15" (default), "Tu15XUV", "Ribas05XUV", "1e-3", "1e-3XUV",
                 "Johnstone20", "Johnstone20XUV"
    
    EUV_relation (str): "Linsky" (default), or: "Johnstone", "Chadney",
                                                "SanzForcada"
    ML_rel (str): "ZAMS_Thomas" (default), or: "MS_Mamajeck"
    Returns:
    --------
    L_HE (float): X/XUV luminosity in erg/s
    """
    
    if ML_rel == "MS_Mamajeck":
        Lbol = 10**mass_luminosity_relation(mass_star)
    elif ML_rel == "ZAMS_Thomas":
        Lbol = ZAMS_mass_luminosity_relation(mass_star)
    
    if paper == "OwWu17XUV":
        return l_xuv_OwWu17(t_, mass_star)
    elif paper == "WaDa18XUV":
        return l_xuv_WaDa18(t_, mass_star)
    elif paper == "Kuby20XUV":
        return l_xuv_Kubyshkina20(t_, Lbol)
    elif paper == "LoRi17XUV":
        return l_xuv_LopezRice17(t_, Lbol)
    elif paper == "Ribas05XUV":
        return l_xuv_Ribas05(t_, Lbol)
    elif paper == "Tu15":
        return 10**(-3.13) * Lbol * const.L_sun.cgs.value
    elif paper == "Tu15XUV":
        Lx_sat = 10**(-3.13) * Lbol * const.L_sun.cgs.value
        return l_xuv_all(Lx_sat, EUV_relation, mass_star)
    elif paper == "1e-3":
        return 10**(-3.0) * Lbol * const.L_sun.cgs.value
    elif paper == "1e-3XUV":
        Lx_sat = 10**(-3.0) * Lbol * const.L_sun.cgs.value
        return l_xuv_all(Lx_sat, EUV_relation, mass_star)
    elif paper == "Johnstone20":
        Rx_sat = 5.135 * 1e-4
        return Rx_sat * (Lbol * const.L_sun.cgs.value)
    elif paper == "Johnstone20XUV":
        Lx_sat = (5.135 * 1e-4) * (Lbol * const.L_sun.cgs.value)
        return l_xuv_all(Lx_sat, EUV_relation, mass_star)


def calculate_Lx_sat(mass_star=1.0, paper="Tu15",
                     EUV_relation="Linsky", ML_rel="ZAMS_Thomas"):
    """ Uses l_high_energy to estimate the X-ray (or XUV) saturation luminosity.
    
    paper (str): "OwWu17XUV", "WaDa18XUV", "Kuby20XUV", "LoRi17XUV",
                 "Tu15" (default), "Tu15XUV", "Ribas05XUV", "1e-3", "1e-3XUV",
                 "Johnstone20", "Johnstone20XUV"
                 
    M-L relation:
    - "MS_Mamajeck" - can either be the main-sequence one from Mamajeck
    - "ZAMS_Thomas": or estimate Lbol right when star reaches the ZAMS
                     (from T. Steindl's pre-MS MESA models)
          
    EUV_relation: user has 4 options to estimate the EUV content based on X-ray
    
    Returns:
    --------
    Lx or Lxuv (in erg/s)
    """
    
    return l_high_energy(1.0, mass_star, paper, EUV_relation, ML_rel)
    
    
def l_xuv_OwWu17(t_, mass_star):
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
    L_sat = 10**(-3.5) * (mass_star) * const.L_sun.cgs.value
    a_0 = 0.5
    if t_ < t_sat:
        L_HE = L_sat
    elif t_ >= t_sat:
        L_HE = L_sat * (t_ / t_sat)**(-1.0 - a_0)
    return L_HE # in erg/s
    

def l_xuv_WaDa18(t_, mass_star=1.0):
    """ t_ and t_sat in Myr;
    for solar-like star! (1 M_sun)
    based on Wang & Dai 2018
    SAME AS OwWu17!! """
    t_sat = 100.
    if t_ > t_sat:
        return 10**(-3.5) * const.L_sun.cgs.value * (t_*1e6/1e8)**(-1.5) \
                * (mass_star)
    else:
        return 10**(-3.5) * const.L_sun.cgs.value * (mass_star)

    
def l_xuv_Kubyshkina20(t_, Lbol=1.0):
    """t_ and t_sat in Myr;
    for solar-like star! (1 M_sun)
    based on Ribas 2005"""
    t_sat = 62.
    if t_ > t_sat:
        return 29.7 * (t_/1e3)**(-1.23) * (4*np.pi*(const.au.cgs.value)**2) \
                * Lbol
    else:
        return 886 * (4*np.pi*(const.au.cgs.value)**2) * Lbol


def l_xuv_LopezRice17(t_, L_bol=1.0):
    """
    t_ and t_sat in Myr, L_bol in L_sun;
    based on Ribas 2005 & Jackson 2011;
    for sun-like stars (~ 1-1200 Angstron)"""
    if t_ < 100.:
        return 10**(-3.5) * L_bol * const.L_sun.cgs.value
    else:
        return 29.7 * (L_bol) * (t_/1e3)**(-1.23) \
                * (4*np.pi*(const.au.cgs.value)**2)


def l_xuv_Ribas05(t_, L_bol=1.0, t_sat=100.):
    """ ‘XUV’: 1 - 1200 Angstrom
    
    Johnstone 2020 -> XUV: 0.1 - 92 nm
     - EUV: 10 to 92 nm 
     - X-rays: 2.4 to 0.1 keV (0.517–12.4 nm)
    """
    if t_ > t_sat:
        return 29.7 * (t_/1e3)**(-1.23) * (4*np.pi*(const.au.cgs.value)**2) \
                * L_bol
    else:
        return 29.7 * (t_sat/1e3)**(-1.23) * (4*np.pi*(const.au.cgs.value)**2) \
                * L_bol


def calculate_EUV_flux_Linsky15(flux_Lya):
    """ Linsky 2015"""
    # flux at 1 A.U. (10-117 nm), valid for F5 to K7 V stars.
    log_flux_10_to_20_over_Lya = -1.357 + 0.344 * np.log10(flux_Lya)
    log_flux_20_to_30_over_Lya = -1.300 + 0.309 * np.log10(flux_Lya)
    log_flux_30_to_40_over_Lya = -0.882
    log_flux_40_to_50_over_Lya = -2.294 + 0.258 * np.log10(flux_Lya)
    log_flux_50_to_60_over_Lya = -2.098 + 0.572 * np.log10(flux_Lya)
    log_flux_60_to_70_over_Lya = -1.920 + 0.240 * np.log10(flux_Lya)
    log_flux_70_to_80_over_Lya = -1.894 + 0.518 * np.log10(flux_Lya)
    log_flux_80_to_91_over_Lya = -1.811 + 0.764 * np.log10(flux_Lya)
    log_flux_91_to_117_over_Lya = -1.004 + 0.065 * np.log10(flux_Lya)
    log_flux_Ly_series_over_Lya = -1.798 + 0.351  * np.log10(flux_Lya)
    
    return (10**log_flux_10_to_20_over_Lya + 10**log_flux_20_to_30_over_Lya +\
           10**log_flux_30_to_40_over_Lya + 10**log_flux_40_to_50_over_Lya +\
           10**log_flux_50_to_60_over_Lya +\
           10**log_flux_60_to_70_over_Lya + 10**log_flux_70_to_80_over_Lya +\
           10**log_flux_80_to_91_over_Lya + 10**log_flux_91_to_117_over_Lya + \
           10**log_flux_Ly_series_over_Lya) * flux_Lya


def calculate_EUV_luminosity_Linsky13_15(Lx):
    """ Input: Lx luminosity (erg/s)
        Output: EUV luminosity (erg/s)
        -> L_x to L_Lya relation from Linsky 13 und
           L_Lya to L_EUV conversion from Linsky 15 """
    
    logLya = 19.7 + 0.322 * np.log10(Lx)   
    flux_Lya = 10**logLya / (4 * np.pi * (1.0 * const.au.cgs.value)**2)
    # flux at 1 A.U. (10-117 nm), valid for F5 to K7 V stars.
    flux_EUV = calculate_EUV_flux_Linsky15(flux_Lya)
    L_EUV = flux_EUV * (4 * np.pi * (1.0 * const.au.cgs.value)**2)
    return L_EUV


def M_R_relation(M):
    """ Z. Eker et al. 2018"""
    R = 0.438 * M**2 + 0.479 * M + 0.075
    return R

def R_M_relation(R):
    """ Z. Eker et al. 2018"""
    c = (0.075-R)
    b = 0.479
    a = 0.438
    M1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    M2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    #0 = 0.438 * M**2 + 0.479 * M + (0.075-R_
    if M1 >= 0.: return M1
    else: return M2


def calculate_EUV_luminosity_Chadney15(lx, mstar=None, radius=None):
    """ Function to estimate the EUV suface flux using the scaling relations 
    given by Johnstone et al. (2020), coupled with a M-R relation to estimate
    the total luminosity (Eker et al., 2018). 
    F_x is the stellar surface flux in erg/s/cm^2
    """
    if mstar == None:
        raise Exception("To use the Chadney surface flux relation to estimate" +
                       " the EUV luminosity, you need to specify the mass of" +
                       " the star, e.g. mstar=1.0.")
        
    if radius != None:
        Rstar = radius
    else:
        Rstar = M_R_relation(mstar)
    A_star = 4 * np.pi * (Rstar*const.R_sun.cgs.value)**2
    F_x = lx / A_star
    logF_x = np.log10(F_x)
    LogF_EUV = 2.63 + 0.58*logF_x
    Feuv = 10**LogF_EUV
    return Feuv * A_star


def calculate_EUV_luminosity_Johnstone20(lx, mstar=None, radius=None):
    """ Function to estimate the EUV suface flux using the scaling relations 
    given by Johnstone et al. (2020), coupled with a M-R relation to estimate
    the total luminosity (Eker et al., 2018). 
    F_x is the stellar surface flux in erg/s/cm^2
    """
    if mstar == None:
        raise Exception("To use the Johnstone surface flux relation to estimate" +
                       " the EUV luminosity, you need to specify the mass of" +
                       " the star, e.g. mstar=1.0.")
    if radius != None:
        Rstar = radius
    else:
        Rstar = M_R_relation(mstar)
    A_star = 4 * np.pi * (Rstar*const.R_sun.cgs.value)**2
    F_x = lx / A_star
    logFeuv1 = 2.04 + 0.681 * np.log10(F_x)
    logFeuv2 = -0.341 + 0.920*logFeuv1
    Feuv = 10**logFeuv1 + 10**logFeuv2
    return Feuv * A_star
    
    
def calculate_EUV_luminosity_SanzForcada11(lx):
    """Function to estimate the EUV luminosity using the scaling relations 
    given by Sanz-Forcada et al. (2011); the measured X-ray luminosity can 
    be extrapolated to the total high-energy radiation as given below. 
    To get the combined high-energy luminosity (L_xuv), we add L_x and L_euv.
    THEY USE: Lx(0.1-2.4 keV) ROSAT band
    """
    log_L_euv = 0.860 * np.log10(lx) + 4.80
    return 10.**log_L_euv  # erg/s


def l_xuv_Linsky(lx):
    """Function to estimate the combined XUV (X-ray + EUV) luminosity using the
    scaling relations given by Linsky et al. (2013 & 2015); the measured X-ray
    luminosity can be extrapolated to the total high-energy radiation by first
    using the L_x to L_Lya relation from Linsky 13 and then the L_Lya to L_EUV
    conversion from Linsky 15.
    To get the combined high-energy luminosity (L_xuv), we add L_x and L_EUV.
    
    Input:
    ------
    lx (float): X-ray luminosity (erg/s)
    
    Returns:
    --------
    l_XUV (float): total XUV high-energy luminosity (erg/s)
    """
    
    l_EUV = calculate_EUV_luminosity_Linsky13_15(lx)
    l_XUV = lx + l_EUV
    return l_XUV


def l_xuv_SanzForcada(lx):
    """Function to estimate the EUV luminosity using the scaling relations 
    given by Sanz-Forcada et al. (2011); the measured X-ray luminosity can 
    be extrapolated to the total high-energy radiation as given below. 
    To get the combined high-energy luminosity (L_xuv), we add L_x and L_euv.
    THEY USE: Lx(0.1-2.4 keV) ROSAT band
    """
    log_L_euv = 0.860 * np.log10(lx) + 4.80
    l_EUV = 10.**log_L_euv
    log_L_xuv = np.log10(lx + l_EUV)
    return 10.**log_L_xuv  # erg/s


def l_xuv_all(lx, relation="Linsky", mstar=None, radius=None):
    """Function to estimate the EUV luminosity using the scaling relations 
    given either by Linsky et al. (2013, 2015) OR Sanz-Forcada et al. (2011);
    the measured X-ray luminosity is used to estimate the EUV luminosity and 
    by adding L_x and L_euv we obtain the combined high-energy luminosity (X-ray + EUV).

    Set relation to: 
        - relation="Linsky" (default)
        - relation="Chadney" -> mstar=1.0
        - relation="Johnstone" -> mstar=1.0
        - relation="SanzForcada"
    """
    if relation == "Linsky":
        return l_xuv_Linsky(lx)
    
    elif relation == "Chadney":
        leuv = calculate_EUV_luminosity_Chadney15(lx, mstar, radius=radius)
        return lx + leuv
    
    elif relation == "Johnstone":
        leuv = calculate_EUV_luminosity_Johnstone20(lx, mstar, radius=radius)
        return lx + leuv
    
    else:
        return l_xuv_SanzForcada(lx)