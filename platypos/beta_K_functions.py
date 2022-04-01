import numpy as np
import astropy.units as u
from astropy import constants as const


def beta_calc(M_p, R_p, F_xuv,
              beta_settings,
              distance=None, M_star=None, Lbol_solar=None):
    """ beta & K calculator
    
    Function estimates the beta parameter, which is a correction to the
    planetary absorption radius in XUV, as planet appers larger than in
    optical: beta = R_XUV/R_optical.

    Args:
        M_p (float): mass of the planet in Earth units
        R_p (float): radius mass of the planet in Earth units
        F_xuv (float): XUV flux recieved by planet (cgs)
    
        beta_settings (dict): 
            Depending on the estimation procedure, the dictionary
            has 1, 2 or 3 params.
            1) beta_calc (str): "Salz16" or "Lopez17" or "off"
                        a) approximation from a study by Salz et al. (2016)
                        NOTE from paper: "The atmospheric expansion can be
                        neglected for massive hot Jupiters, but in the range of
                        superEarth-sized planets the expansion causes mass-loss
                        rates that are higher by a factor of four."
                        b) approximation from MurrayClay (2009) or Lopez (2017)
                        c) beta = 1
            2) RL_cut (bool): if R_XUV > R_RL, set R_XUV=R_RL
            3) beta_cut (bool): additional parameter if beta_calc="Salz16";
                        - IF cutoff == True, beta is kept constant at the lower
                        boundary value for planets with gravities lower than
                        the Salz sample
                        - IF cutoff == False, the user agrees to extrapolate
                        this relation beyond the Salz-sample limits.                     
    
        distance (float): semi-major axis of planet (needed for RL and Lopez calc)
        M_star (float): mass of star (needed for RL calc)
        Lbol_solar (float): stellar bolometric luminosity (needed for Lopez calc)

    Returns:
    beta (float): beta parameter
    """
    try: 
        beta_calc = beta_settings["beta_calc"]
    except:
        raise KeyError('Need to specify beta_calc ' +\
                       '-> "Salz16", "Lopez17" or "off".')
    
    if beta_calc == "Salz16":
        beta = beta_Salz16(M_p, R_p, F_xuv,
                           beta_settings,
                           distance=distance, M_star=M_star)
        
    elif beta_calc == "Lopez17":
        beta = beta_Lo17(M_p, R_p, F_xuv,
                         distance, Lbol_solar,
                         beta_settings,
                         M_star=M_star)
        
    elif beta_calc == "off":
        beta = 1.
    
    return beta


def beta_Salz16(M_p, R_p, F_xuv,
                beta_settings,
                distance=None, M_star=None):
    """ Calculate Salz beta (see function beta_calc() for more info). """
    try:
        beta_calc = beta_settings["beta_calc"]
        beta_cut = beta_settings["beta_cut"]
        RL_cut = beta_settings["RL_cut"]
    except:
        raise KeyError("For Salz16 beta, specify 'beta_calc', " + \
                       "'beta_cut' and 'RL_cut'!")
    if RL_cut == True:
        if distance == None:
            raise KeyError('RL_cut=True, but distance not specified.')
        if M_star == None:
            raise KeyError('RL_cut=True, but M_star not specified.')
    
    M_EARTH = 5.972364730419773e+27 #const.M_earth.cgs.value
    R_EARTH = 637810000.0 #const.R_earth.cgs.value
    G_CONST = 6.674079999999999e-08 #const.G.cgs.value
    M_SUN = 1.9884754153381438e+33 #const.M_sun
    AU = 14959787070000.0 #const.au.cgs.value
    GRAV_POT_MIN = -10**12. # lower limit of Salz-sample

    # calculate Salz beta
    grav_pot = -G_CONST * (M_p * M_EARTH) / (R_p * R_EARTH)
    if (beta_cut == False):
        log_beta = max(0.0, -0.185 * np.log10(-grav_pot)
                            + 0.021 * np.log10(F_xuv) + 2.42)
    
    # if grav. potential outside of Salz range, truncate beta
    # Salz sample spans wide F_xuv range, but not grav. pot.
    elif (beta_cut == True):
        if (np.log10(-grav_pot) < np.log10(-GRAV_POT_MIN)):
            log_beta = max(0.0, -0.185 * np.log10(-GRAV_POT_MIN)
                                + 0.021 * np.log10(F_xuv) + 2.42)
        else: 
            log_beta = max(0.0, -0.185 * np.log10(-grav_pot)
                            + 0.021 * np.log10(F_xuv) + 2.42)
    
    beta = 10**log_beta

    # if R_XUV is larger than the planetary Roche lobe radius: R_XUV == R_RL 
    if RL_cut == True:
        R_RL = distance * AU * (M_p * M_EARTH / \
                                (3. * (M_p * M_EARTH + \
                                       M_star * M_SUN)))**(1./3)
        R_RL = (R_RL / R_EARTH) # convert to Earth radii
        if (beta * R_p) > R_RL:
            # if R_XUV > R_Rl, calcualte beta such that R_XUV=R_RL
            beta = R_RL / R_p
 
    return beta


def beta_Lo17(M_p, R_p, F_xuv,
              distance, Lbol_solar,
              beta_settings,
              M_star=None):
    """ Calculate Lopez beta (see function beta_calc() for more info). """
    try:
        beta_calc = beta_settings["beta_calc"]
        RL_cut = beta_settings["RL_cut"]
    except:
        raise KeyError("For Lopez17 beta, specify 'beta_calc' and 'RL_cut'!")
    
    if beta_calc == "Lopez17":
        if distance == None:
            raise KeyError('distance not specified.' \
                            + 'Needed for Lopez17 beta estimation!')
        if Lbol_solar == None:
            raise KeyError('Lbol_solar not specified.' \
                            + 'Needed for Lopez17 beta estimation!')
    
    if RL_cut == True:
        if distance == None:
            raise KeyError('RL_cut=True, but distance not specified.')
        if M_star == None:
            raise KeyError('RL_cut=True, but M_star not specified.')
    
    M_HY = 1.673532836356e-24 #const.m_p.cgs.value + const.m_e.cgs.value
    M_EARTH = 5.972364730419773e+27 #const.M_earth.cgs.value
    R_EARTH = 637810000.0 #const.R_earth.cgs.value
    G_CONST = 6.674079999999999e-08 #const.G.cgs.value
    M_SUN = 1.9884754153381438e+33 #const.M_sun.cgs.value
    AU = 14959787070000.0 #const.au.cgs.value
    SIGMA_SB = 5.6703669999999995e-05 #const.sigma_sb.cgs.value
    L_SUN = 3.828e+33 #const.L_sun.cgs.value
    K_B = 1.38064852e-16 #const.k_B.cgs.value
    EV = 1.6021766208000004e-12 #(1*u.eV).cgs.value
    BAR = 1000000.0 #(1.*u.bar).cgs.value 
    
    rho_p = ((M_p * M_EARTH) / (4./3 * np.pi * (R_p * R_EARTH)**3))
    
    # Estimate R_XUV as done in Lopez 17 
    surf_grav = (G_CONST * (M_p * M_EARTH)) / (R_p * R_EARTH)**2
    mu_below = 2.5  # for H/He envelopes
    # scaleheight in regime between optical and XUV photosphere
    t_eq = (((Lbol_solar * L_SUN) / (16 * np.pi * SIGMA_SB * \
                                     (distance * AU)**2))**(0.25))
    H_below = (K_B * t_eq) / (mu_below * M_HY * surf_grav)
    
    # following Murray-Clay 2009, estimate the pressure at the tau_XUV=1 boundary
    # (pressure at base of wind/ XUV photosphere) from the photo-ionization of H
    P_photo = (20. * 1e-3) * BAR
    h_nu0 = 20. * EV #typical EUV (XUV) energy (not integrated over whole spectrum!)
    sigma_nu0 = 6.0*1e-18 * (h_nu0 / (13.6 * EV))**(-3) #cm**2
    P_base = (M_HY * surf_grav) / sigma_nu0
    R_base = (R_p * R_EARTH) + H_below * np.log(P_photo / P_base)
        
    beta = (R_base / R_EARTH) / R_p

    if RL_cut == True:
        R_RL = distance * AU * (M_p * M_EARTH / \
                                (3.*(M_p * M_EARTH + M_star * M_SUN)))**(1./3)
        R_RL = (R_RL / R_EARTH) # convert to Earth radii
        if (beta * R_p) > R_RL:
            # if R_XUV > R_Rl, calcualte beta such that R_XUV=R_RL
            beta = R_RL / R_p
    
    return beta

   
def K_fct(a_pl, M_pl, M_star, R_pl):
    """ K correction factor to take into account tidal influences of the
    host star on the planetary atmosphere (from Erkaev et al. 2007).

    Parameters:
    -----------
    a_pl (float): star-planet separation in A.U.
    M_pl (float or list): mass of the planet in Earth units
    M_star (float): mass of the star in solar units
    R_pl (float): radius of the planet in Earth units

    Returns:
    --------
    K (float) or K (array)
    """
              
    M_EARTH = 5.972364730419773e+27 #const.M_earth.cgs.value
    R_EARTH = 637810000.0 #const.R_earth.cgs.value
    M_SUN = 1.9884754153381438e+33 #const.M_sun.cgs.value
    AU = 14959787070000.0 #const.au.cgs.value

    if (type(M_pl) == float) or (type(M_pl) == np.float64):
        # if M_pl is single value
        R_roche = (a_pl * AU) * ((M_pl * M_EARTH) / \
                                 (3. * (M_star * M_SUN)))**(1./3)
        K = 1. - 3. * (R_pl * R_EARTH) / (2. * R_roche) \
               + (R_pl * R_EARTH)**3. / (2. * R_roche**3.)
        
        return K

    elif len(M_pl) > 1:
        # if F_xuv is array
        Ks = []
        for i in range(len(M_pl)):
            R_roche_i = (a_pl[i] * AU) * ((M_pl[i] * M_EARTH) / \
                                          (3. * (M_star[i] * M_SUN)))**(1./3)
            K_i = 1. - 3. * (R_pl[i] * R_EARTH) / (2. * R_roche_i) \
                  + (R_pl[i] * R_EARTH)**3. / (2. * R_roche_i**3.)
            Ks.append(K_i)
        Ks = np.array(Ks)
        
        return Ks


def beta_fct(M_p, F_xuv, R_p, cutoff=True):
    """
    OUTDATED!!!! Now I have a much better function called beta_calc()
    
    Function estimates the beta parameter, which is a correction to the
    planetary absorption radius in XUV, as planet appers larger than in
    optical; this approximation comes from a study by Salz et al. (2016)
    NOTE from paper: "The atmospheric expansion can be neglected for massive
    hot Jupiters, but in the range of superEarth-sized planets the expansion
    causes mass-loss rates that are higher by a factor of four."
    
    - if cutoff == True, beta is kept constant at the lower boundary value
    for planets with gravities lower than the Salz sample
    - if cutoff == False, the user agrees to extrapolate this relation beyond
    the Salz-sample limits.

    Parameters:
    ----------
    M_p (float): mass of the planet in Earth units
    F_xuv (float or list): XUV flux recieved by planet
    R_p (float): radius mass of the planet in Earth units

    Returns:
    --------
    beta (float or array): beta parameter
    """

    M_EARTH = const.M_earth.cgs.value
    R_EARTH = const.R_earth.cgs.value
    GRAV_POT_MIN = -10**12. # lower limit of Salz-sample

    if (type(F_xuv) == float) or (type(F_xuv) == np.float64):
        # if F_xuv is single value
        grav_pot = -const.G.cgs.value * (M_p*M_EARTH) / (R_p*R_EARTH)
        log_beta = max(0.0, -0.185 * np.log10(-grav_pot)
                            + 0.021 * np.log10(F_xuv) + 2.42)
        beta = 10**log_beta
        
        if (cutoff == True) and (np.log10(-grav_pot) < np.log10(-GRAV_POT_MIN)):
            log_beta = max(0.0, -0.185 * np.log10(-GRAV_POT_MIN)
                            + 0.021 * np.log10(F_xuv) + 2.42)
            beta = 10**log_beta

        return beta

    elif len(F_xuv) > 1:
        # if F_xuv is a list
        betas = []
        for i in range(len(F_xuv)):
            grav_pot_i = -const.G.cgs.value \
                         * (M_p[i]*M_EARTH) / (R_p[i]*R_EARTH)
            log_beta_i = max(0.0, -0.185 * np.log10(-grav_pot_i)
                                  + 0.021 * np.log10(F_xuv[i]) + 2.42)
            beta_i = 10**log_beta_i
            
            if (cutoff == True) and \
                (np.log10(-grav_pot_i) < np.log10(-GRAV_POT_MIN)):
                log_beta_i = max(0.0, -0.185 * np.log10(-GRAV_POT_MIN)
                                      + 0.021 * np.log10(F_xuv[i]) + 2.42)
                beta_i = 10**log_beta_i
            
            betas.append(beta_i)
            
        betas = np.array(betas)
        return betas