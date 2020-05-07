# this file contains functions related to Lx:
#
# test
#

####################################################################################################################

import numpy as np
import astropy.units as u
from astropy import constants as const

# re-write this function to pass a dictionary

def Lx_evo(t, track_dict):
    """
    Function to calculate the X-ray luminosity, Lx, of the host star at a given time t.
    
    NOTE: I tried to set up this function such that it works going 
    forward (for young system) and backwards (for older system) in time;
    -> so t_curr(ent) is a little confusing! -> 2 we have two cases!!
    
    Parameters Case 1 - a young system which you want to evolve forward in time 
    (i.e. estimate the mass loss the planet will undergo in the future):
    ---------------------------------------------------------------------------
    @param t_start: (float) 
        Starting time for the forward simulation in Myr (e.g. 23 Myr for V1298 Tau)
    @param Lx_max: (float)
        Measured Lx value at the system age (i.e. t_start)
    @param t_curr: (1-d numpy array or integer) 
        Set to 1 Gyr (with Lx_curr = Lx at 1 Gyr), which is where, by design, the three 
        activity tracks converge (based on Tu et al.(2015)/ Johnstone  et al.(2015) models); 
    @param Lx_curr
    @param t_5Gyr: (float)
        Defines, together with Lx_5Gyr, the slope common for all three 
        activity evolution tracks past 1 Gyr (also based on Tu et al.(2015));
    @param Lx_5Gyr: (float)
        Defines, together with t_5Gyr, the slope common for all three 
        activity evolution tracks past 1 Gyr (also based on Tu et al.(2015));
    
    Parameters Case 2 - an older system, which you want to evolve backwards in time 
    (i.e. estimate the mass loss the planet has undergone until now):
    ---------------------------------------------------------------------------
    @param t_start is the youngest age you want to calculate backwards 
    (i.e. this shouldbe sth. like ~10 Myr, or close to the disk clearing timescale); 
    @param Lx_max: (float) Some saturation X-ray luminosity (e.g. Lx,sat/Lbol ~ 10^(-3)),
    @param t_curr: (float) Needs to be set to the system age (e.g. t_curr=700 Myr for K2-136b); 
    @param Lx_curr: (float) If a measured Lx value is available for the system, otherwise need to estimate;
    @param t_5Gyr: (float)
        Defines, together with Lx_5Gyr, the slope common for all three 
        activity evolution tracks past 1 Gyr (also based on Tu et al.(2015));
    @param Lx_5Gyr: (float)
        Defines, together with t_5Gyr, the slope common for all three 
        activity evolution tracks past 1 Gyr (also based on Tu et al.(2015));
    
    NOTE: t_5Gyr (and Lx_5Gyr) could be changed to create whatever slope desired (for a future evolution).
    Sidenote: this function is not very pretty, needs improvements...
    
    
    
    @return Lx_at_time_t: (float)
        Pass one t-value, and get the corresponding Lx
    """
    
    # read in the parameters from the provided dictionary -> these are all the parameters required to define the tracks
    t_start, t_sat, t_curr, t_5Gyr = track_dict["t_start"], track_dict["t_sat"], track_dict["t_curr"], track_dict["t_5Gyr"]
    Lx_max, Lx_curr, Lx_5Gyr = track_dict["Lx_max"], track_dict["Lx_curr"], track_dict["Lx_5Gyr"]
    dt_drop, Lx_drop_factor = track_dict["dt_drop"], track_dict["Lx_drop_factor"]
    
    ########################################################################################################################
    
    # Define function for calculating a power law
    powerlaw = lambda x, k, index: k * (x**index)
    
    # t_curr has to be greater than t_start
    if (t_curr < t_start) or (t < t_start):
        
        if t <= t_start: # this allows me (for the V1298 Tau system) to produce a saturation regime before 23 Myr with Lx_max
            Lx = Lx_max
        else:
            print("Make sure t_curr > t_start, t >= t_start")
    
    elif t > t_curr:
        # this is the regime past 1 Myr
        # I use the Tu 2015 slope which is approx the same for all three evolutionary tracks (and based on current solar value);
        # this is mainly for consistency since we approximate our tracks based on the Tu tracks
        alpha = (np.log10(Lx_5Gyr/Lx_curr))/(np.log10(t_5Gyr/t_curr))
        k = 10**(np.log10(Lx_curr) - alpha*np.log10(t_curr))
        Lx = powerlaw(t, k, alpha)
        return Lx
    
    elif t_curr > t_start:
        ###################################################################################
        # dt_drop==0 means we create a track with only the saturation regime and 
        # a drop to the converging Lx at 1 Gyr
        ###################################################################################
        if dt_drop==0: # then t_sat == t_drop
            if t_start >= t_sat: # then t_sat <= t_start < t_curr
                #print("t_sat <= t_start < t_curr")
                if t >= t_sat: #and t <= t_curr:
                    # only data in power law regime
                    alpha = (np.log10(Lx_curr/Lx_max))/(np.log10(t_curr/t_sat))
                    k = 10**(np.log10(Lx_max) - alpha*np.log10(t_sat))
                    Lx = powerlaw(t, k, alpha)
                    #print(Lx)
                    
            elif t_start < t_sat: # then t_start < t_sat < t_curr
                if t > t_sat: #and t <= t_curr:
                    alpha = (np.log10(Lx_curr/Lx_max))/(np.log10(t_curr/t_sat))
                    k = 10**(np.log10(Lx_max) - alpha*np.log10(t_sat))
                    Lx = powerlaw(t, k, alpha)
                    
                elif t <= t_sat:
                    Lx = Lx_max
                    
        elif dt_drop > 0: # then t_sat != t_drop\
            ###################################################################################
            # dt_drop>0 means we create a more fancy track with the saturation regime and then 
            # a drop with slope change to the converging Lx at 1 Gyr
            ###################################################################################
            t_drop = t_sat + dt_drop # t_sat < t_drop
            
            if t <= t_sat:
                Lx = Lx_max
            elif (t > t_sat) and (t <= t_drop): # first of the two slopes
                alpha_drop1 = (np.log10((Lx_max/Lx_drop_factor)/Lx_max))/(np.log10(t_drop/t_sat))
                k_drop1 = 10**(np.log10(Lx_max) - alpha_drop1*np.log10(t_sat))
                Lx = powerlaw(t, k_drop1, alpha_drop1)
                
            elif t > t_drop:
                alpha_drop2 = (np.log10(Lx_curr/(Lx_max/Lx_drop_factor)))/(np.log10(t_curr/t_drop))
                k_drop2 = 10**(np.log10((Lx_max/Lx_drop_factor)) - alpha_drop2*np.log10(t_drop))
                Lx = powerlaw(t, k_drop2, alpha_drop2)     
    return Lx

#####################################################################################################################
# define flux
def flux_at_planet_earth(L, a_p):
    """Function calculates the flux that the planet recieves at a given distance normalized with Earth's flux
    -> which I approximate with the semi-major axis 
    (i.e. I ignore the eccentricity of the planetary orbits)"""
    A = (4.*np.pi*(a_p*const.au.cgs)**2)
    F = (L*u.erg/u.s)/A
    flux_earth = 1373*1e7*(u.erg/u.s)/(100*u.cm*100*u.cm)
    F_earth = (F/flux_earth).value
    return F_earth

#####################################################################################################################
# define flux
def flux_at_planet(L, a_p):
    """Function calculates the flux that the planet recieves at a given distance -> which I approximate with the semi-major axis 
    (i.e. I ignore the eccentricity of the planetary orbits)"""
    A = (4.*np.pi*(a_p*const.au.cgs)**2)
    F = (L*u.erg/u.s)/A
    return F
#####################################################################################################################
def L_xuv_all(Lx):
    """Function to estimate the EUV luminosity using the scaling relations given by Sanz-Forcada et al. (2011);
    the measuredradius_planet_volatilety can be extrapolated to the total high-energy radiation as given below. 
    To get the combined high-energy flux, we add L_x and L_euv to get L_xuv.
    THEY USE: Lx(0.1-2.4 keV) ROSAT band
    (LX and LEUV are the X-ray (λλ 5−100 Å) and EUV (λλ 100−920 Å) luminosities, in erg s−1.)
    """
    log_L_euv = 0.860*np.log10(Lx) + 4.80
    Leuv = 10.**log_L_euv
    log_L_xuv = np.log10(Lx+Leuv)
    return 10.**log_L_xuv#*u.erg/u.s


#####################################################################################################################
#####################################################################################################################
def Lx_relation_Booth(t, R_star):
    """R_star in terms of solar units"""
    log_Lx = 54.65 - 2.8*np.log10(t) + 2*np.log10(R_star)
    return 10.**log_Lx

#####################################################################################################################
def calculate_Lx_sat(L_star_bol):
    """*NOTE*: it is not clear yet, if L_x is directly related to L_bol 
    -> or rather that the size of star (radius) is correlated with the 
    number of active regions, i.e. bigger surface, bigger L_bol, brighter L_x"""
    return 10**(-3)*(L_star_bol*const.L_sun.cgs).value