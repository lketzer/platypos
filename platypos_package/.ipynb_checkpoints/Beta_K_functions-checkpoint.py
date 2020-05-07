# beta and K functions
import astropy.units as u
from astropy import constants as const
import numpy as np

# rename to beta_fct & K_fct

def beta_fct_LO14(M_p, F_xuv, R_p):
    """Function estimates the beta parameter, which is a correction to the planetary absorption radius in XUV 
    -> planet appers larger than in optical; this approximation is used in Lalitha et al. 2018 and comes 
    originally from Salz M., Schneider P., Czesla S., Schmitt J., 2016a, Astronomy & Astrophysics, 585, L2
    -> NOTE from paper: 'The atmospheric expansion can be neglected for massive hot Jupiters, but in the range 
    of superEarth-sized planets the expansion causes mass-loss rates that are higher by a factor of four.' 
    Sidenote: fct works for arrays of M_p/Fxuv or single values.
    
    Parameters:
    ----------
    @param M_p: (float) An float for input; current mass of the planet
    @param F_xuv: (float)
    @param R_p: (float)
    """
    M_earth = const.M_earth.cgs.value
    R_earth = const.R_earth.cgs.value
    
    if (type(F_xuv)==float) or (type(F_xuv)==np.float64): # if F_xuv is single value
        grav_pot = -const.G.cgs.value * (M_p*M_earth) / (R_p*R_earth)
        log_beta = max(0.0, -0.185 * np.log10(-grav_pot) + 0.021 * np.log10(F_xuv) + 2.42)
        beta = 10**log_beta
        #print("beta = ", beta)
        return beta
    elif len(F_xuv) > 1: # if F_xuv is array
        betas = []
        for i in range(len(F_xuv)):
            grav_pot_i = -const.G.cgs.value * (M_p[i]*M_earth) / (R_p[i]*R_earth)
            log_beta_i = max(0.0, -0.185 * np.log10(-grav_pot_i) + 0.021 * np.log10(F_xuv[i]) + 2.42)
            beta_i = 10**log_beta_i
            betas.append(beta_i)
        betas = np.array(betas)
        return betas

def K_fct_LO14(a_pl, M_pl, M_star, R_pl):
    """K correction factor to take into account tidal influences of the host star on the planetary atmosphere (from Erkaev et al. 2007)
    Sidenote: fct works for arrays of M_p/Fxuv or single values.
    
    Parameters:
    ----------
    a_pl:
    M_pl:
    M_star:
    R_pl:
    """
    AU = const.au.cgs.value
    M_sun = const.M_sun.cgs.value
    M_earth = const.M_earth.cgs.value
    R_earth = const.R_earth.cgs.value
    
    #print("M_pl = ", M_pl)
    if (type(M_pl)==float) or (type(M_pl)==np.float64): # if M_pl is single value
        R_roche = (a_pl*AU) * ((M_pl*M_earth)/(3*(M_star*M_sun)))**(1./3)
        K = 1. - 3.*(R_pl*R_earth)/(2.*R_roche) + (R_pl*R_earth)**3./(2.*R_roche**3.)
        #print("K = ", K)
        return K
    elif len(M_pl) > 1: # if F_xuv is array
        Ks = []
        for i in range(len(M_pl)):
            R_roche_i = (a_pl*AU)*((M_pl[i]*M_earth)/(3*(M_star*M_sun)))**(1./3)
            K_i = 1. - 3.*(R_pl[i]*R_earth)/(2.*R_roche_i) + (R_pl[i]*R_earth)**3./(2.*R_roche_i**3.)
            Ks.append(K_i)
        Ks = np.array(Ks)
        return Ks
