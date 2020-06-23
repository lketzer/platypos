import numpy as np
import math
import sys
import astropy.units as u
from astropy import constants as const

from Lx_evo_and_flux import Lx_evo
from Lx_evo_and_flux import flux_at_planet_earth
from Lx_evo_and_flux import L_xuv_all
from Lx_evo_and_flux import flux_at_planet
import Planet_models_LoFo14 as plmoLoFo14
import Planet_model_Ot20 as plmoOt20
import Beta_K_functions as bk
from Mass_loss_rate_function import mass_loss_rate_forward_LO14
from Mass_loss_rate_function import mass_loss_rate_forward_Ot20


def mass_planet_RK4_forward_LO14(epsilon, K_on, beta_on, planet_object,
                                 initial_step_size, t_final, track_dict):
    """USED: 4th order Runge-Kutta as numerical integration  method 
    Integrate from the current time (t_start (where planet has R0 and M0) into
    the future taking into account photoevaporative mass loss. 

    Parameters:
    -----------
    epsilon (float): evaporation efficiency
    K_on (str): set use of K parameter on or off ("on" or "off)
    beta_on (str): set use of beta parameter on or off ("on" or "off)
    planet_object: object of planet class which contains also stellar parameters
                   and info about stellar evo track
    step_size (float): initial step_size, variable
    t_final (float): final time of simulation
    track_dict (dict): dictionary with Lx evolutionary track parameters
    
    [NOTE: the implementation of a variable step size is somewhat preliminary.
    The step size is adjusted (made smaller or bigger depending how fast or
    slow the mass/radius changes) until the final time step greater than
    t_final. This means that if the step size in the end is e.g. 10 Myr, and
    the integration is at 4999 Myr, then last time entry will be 4999+10 ->
    5009 Myr.]

    Returns:
    --------
    t_arr (array): time array to trace mass and radius evolution
    M_arr (array): mass array with mass evolution over time (mass decrease)
    R_arr (array): radius array with radius evolution over time (from thermal
                   contraction and photoevaporative mass-loss)
    Lx_arr (array): array to trace the X-ray luminosity (mainly for consistency
                    checks)
    """ 
    # define some constants
    M_earth = const.M_earth.cgs.value
    R_earth = const.R_earth.cgs.value
    Myr_to_sec = 1e6*365*86400
    
    # initialize the starting values for the X-ray luminosity at t_start, the 
    # starting planet parameters, as well as beta & K (at t_start);
    # convert X-ray to XUV lum. and calculate planet's high energy incident flux
    Lx0 = Lx_evo(t=track_dict["t_start"], track_dict=track_dict)
    Lxuv0 = L_xuv_all(Lx0)
    Fxuv0 = flux_at_planet(Lxuv0, planet_object.distance)

    f_env_0 = f_env = planet_object.fenv
    R0 = R = planet_object.radius
    M0 = M = planet_object.mass
    rho0 = rho = plmoLoFo14.density_planet(M0, R0)
    M_env0 = M_env = M0 - planet_object.core_mass
    M_core = planet_object.core_mass
    R_core = planet_object.core_radius
    # CRITERION for when the planet has lost all atmosphere
    # for the LoFo14 planets the core mass and thus the core radius is fixed.
    # So when the planet mass gets smaller or equal to the core mass, we assume
    # only the bare rocky core is left

    if beta_on == "yes":
        beta = beta0 = bk.beta_fct(M0, Fxuv0, R0)
    elif beta_on == "no":
        beta = beta0 = 1.
    if K_on == "yes":
        K = K0 = bk.K_fct(planet_object.distance, M0,
                          planet_object.mass_star, R0)
    elif K_on == "no":
        K = K0 = 1.

    # since the step size is adaptive, I use lists to keep track of the time,
    # mass, radius and Lx evolution
    M_arr = []
    M_arr.append(M0)
    R_arr = []
    R_arr.append(R0) 
    t_arr = []
    t0 = t = track_dict["t_start"]
    t_arr.append(t0) 
    Lx_arr = []
    Lx_arr.append(Lx0)

    dt = initial_step_size
    # NOTE: minimum and maximum step size are HARDCODED for now (see further 
    # down in code for more details)
    min_step_size, max_step_size = 1e-2,  10.

    i = 1  # counter to track how many traced RK iterations have been performed
    envelope_left = True  # variable to flag a planet if envelope is gone
    close_to_evaporation = False  # variable to flag if planet is close to
                                  # complete atmospheric removal

    while t <= t_final:
        #print("i - t - dt: ", i, t, dt)

        # This step (Lx(t) calculation) is just for me to trace Lx and check if
        # it is correct. It is NOT required since the Lx(t) calculation is 
        # embedded in the mass_loss_rate_fancy function)
        Lx_i = Lx_evo(t=t, track_dict=track_dict)

        # IMPORTANT points on the time step:
        # When the initial time step is too large OR the planet mass becomes
        # very close to the core mass (after several time steps), it can happen
        # that one of the RK substeps leads to such a large mass lost that the
        # new planet mass is smaller than the core mass.
        # Distinguish between two cases:
        # 1) initial time step is too large such that M_lost = nan after the
        # first iteration (i.e. Rk substep mass < core mass)
        # -> immediately switch to lowest possible step size and let code run
        # from there (i.e. code will make step size bigger again if necessary)
        # 2) at the end of planet evolution when the planet mass gets very close
        # to the core mass, at some point the mass lost is larger than the
        # renmaining atmosphere mass (either at the end of a coplete RK
        # iteration, or this already happends in one of the RK substeps, in
        # which case the mass lost after the complete RK step evaluates to nan
        # and no new planet radius can be calculated). In both cases the planet
        # is assumed to be fully evaporated at t_i + dt.

        while (envelope_left == True):
            # go through RK iterations as long as there is envelope left
            
            # apply Runge Kutta 4th order to find next value of M_dot
            # NOTE: the mass lost in one timestep is in Earth masses
            Mdot1 = mass_loss_rate_forward_LO14(t, epsilon, K_on,
                                                beta_on, planet_object,
                                                f_env, track_dict)
            k1 = (dt*Myr_to_sec * Mdot1)/M_earth 
            M_05k1 = M + 0.5 * k1 # mass after 1st RK step
            M_env_05k1 = M_05k1 - M_core
            f_env_05k1 = (M_env_05k1/M_05k1) * 100 # new envelope mass fraction  
            if (i == 1) and (M_05k1 < M_core):
                # then I am still in the first RK iteration, and the initial
                # step size was likely too large -> set step size to minumum
                dt = min_step_size
                break
            
            Mdot2 = mass_loss_rate_forward_LO14(t+0.5*dt, epsilon, K_on,
                                                beta_on, planet_object,
                                                f_env_05k1, track_dict)
            k2 = (dt*Myr_to_sec * Mdot2)/M_earth
            M_05k2 = M + 0.5 * k2
            M_env_05k2 = M_05k2 - M_core
            f_env_05k2 = (M_env_05k2/M_05k2) * 100
            if (i == 1) and (M_05k2 < M_core):
                # then I am still in the first RK iteration, and the initial
                # step size was likely too large -> set step size to minumum
                dt = min_step_size
                break

            Mdot3 = mass_loss_rate_forward_LO14(t+0.5*dt, epsilon, K_on,
                                                beta_on, planet_object,
                                                f_env_05k2, track_dict)
            k3 = (dt*Myr_to_sec * Mdot3)/M_earth
            M_k3 = M + k3
            M_env_k3 = M_k3 - M_core
            f_env_k3 = (M_env_k3/M_k3) * 100
            if (i == 1) and (M_k3 < M_core):
                # then I am still in the first RK iteration, and the initial
                # step size was likely too large -> set step size to minumum
                dt = min_step_size
                break

            Mdot4 = mass_loss_rate_forward_LO14(t+dt, epsilon, K_on,
                                                beta_on, planet_object,
                                                f_env_k3, track_dict)
            k4 = (dt*Myr_to_sec * Mdot4)/M_earth
            # total mass lost after time-step dt
            M_lost = (k1 + 2*k2 + 2*k3 + k4)/6. 

            # update next value of the planet mass
            M_new = M + M_lost
            M_env_new = M_new - M_core

            # now it is time to check, if atmosphere is gone or 
            # if planet is close to complete atmosphere removal
            if ((np.isnan(M_lost) == True) or (M_new <= M_core))\
                and (dt == min_step_size):
                # if M_lost = nan (i.e. planet evaporates in one of the RK
                # steps) OR the four RK steps finish and the new planet mass is
                # smaller or equal to the core mass, then the planet counts as
                # evaporated!
                # if this condition is satisfied and the step size is already at
                # a minimum, then we assume the current RK iteration would
                # remove all atmosphere and only the rocky core is left at
                # t_i+dt; this terminates the code and returns the final planet
                # properties

                # print("Atmosphere has evaportated! Only bare rocky core left!\
                #        STOP this madness!")

                # since the the stop criterium is reached, we assume at t_i+1
                # the planet only consists of the bare rocky core with the
                # planet mass equal to the core mass and the planet radius equal
                # to the core radius
                t_arr.append(t_arr[-1]+dt)
                M_arr.append(M_core)
                R_arr.append(R_core)
                Lx_arr.append(Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))
                envelope_left = False  # set flag for complete env. removal
                break
                
            elif ((np.isnan(M_lost) == True) or (M_new <= M_core))\
                 and (dt > min_step_size) and (close_to_evaporation == False):
                # planet close to evaporation, but since the step size is not
                # minimum yet, we set it to its minimum value and run the RK
                # iteration again (until above stopping condition is fulfilled)
                dt = min_step_size
                close_to_evaporation = True 
                # this variable is for making sure the code does not run into an
                # infinite loop when the planet is close to evaporation. Once
                # this condition is set to True, the code continues with a fixed
                # minimum step size and it no longer allowed to adjust it.
                break
            
            # this part is new compared to the one used in the PAPER (there we
            # used a fixed step size!)
            # if you're still in the while loop at this point, then calculate
            # new radius and check how drastic the radius change would be;
            # adjust the step size if too drastic or too little
            f_env_new = (M_env_new/M_new)*100 # in %
            R_new = plmoLoFo14.calculate_planet_radius(M_core, f_env_new, t,
                                                   flux_at_planet_earth(
                                                       planet_object.Lbol,
                                                       planet_object.distance),
                                                   planet_object.metallicity)

            # only adjust step size if planet is not close to complete evaporat.
            if (close_to_evaporation == False):
                # then do the check on how much the radius changes
                # R(t_i) compared to R(t_i+dt);
                # if radius change is larger than 1%, make step size smaller
                # by factor 10 OR if radius change is smaller than 0.01%, make
                # step size bigger by factor 10, but not bigger or smaller than
                # max. or min. step size!
                # if radius change too much/little, do not write anything to
                # file, instead do RK iteration again with new step size

                R_change = abs((R-R_new)/R)*100 # radius change compared to
                                                # previous radius - in percent
                if (R_change > 0.1) and (t < track_dict["t_curr"])\
                    and (dt > min_step_size):
                    dt = dt / 10.
                    break

                #elif ((R_change < 1.) & (R_change >=0.1)) & (t < track_dict["t_curr"]) & (dt > min_step_size):
                #    dt = dt / 10.
                #    break

                elif (R_change < (0.01)) and (t < track_dict["t_curr"])\
                     and (dt < max_step_size):
                    dt = dt * 10.
                    break

                # in principle I can  adjust the code such that these hardcoded
                # parameters are different for early planet evolution where
                # much more is happening typically, and late planet evolution
                # where almost no change is occurring anymore
                elif (R_change > 0.1) and (t >= track_dict["t_curr"])\
                     and (dt > min_step_size):
                    dt = dt / 10.
                    break

                elif (R_change < (0.01)) and (t >= track_dict["t_curr"])\
                     and (dt < max_step_size):
                    dt = dt * 10
                    break

                else: # if radius change is ok
                    # sanity check: is new planet mass is still greater than the
                    # core mass? -> then there is still some atmosphere left
                    # in this case update params and go into next RK iteration
                    if ((M + M_lost) - M_core) > 0: 
                        M = M + M_lost  # new planet mass (M_lost is negative)
                        t = t_arr[-1] + dt  # updated time value t_i_plus_1
                        M_arr.append(M)
                        t_arr.append(t)
                        Lx_arr.append(Lx_evo(t=t, track_dict=track_dict))

                        # calculate new envelope mass fraction:
                        M_env = M - M_core
                        f_env = (M_env/M)*100 # in %

                        # calculate new radius with new planet mass/envelope
                        # mass fraction & one time step later
                        R = plmoLoFo14.calculate_planet_radius(M_core, f_env, t,
                                                           flux_at_planet_earth(
                                                             planet_object.Lbol,
                                                             planet_object.distance),
                                                           planet_object.metallicity)
                        R_arr.append(R)
                        i = i+1 # update step to i+1

                    else:
                        # this should never happen
                        sys.exit('sth went wrong of you see this!')
                break
            
            elif (close_to_evaporation == True): 
                # if this consition is true, do not adjust step size
                # based on the radius change
                if ((M + M_lost) - M_core) > 0:
                    M = M + M_lost # new planet mass (M_lost is negative)
                    t = t_arr[-1] + dt #  updated time value t_i_plus_1
                    M_arr.append(M)
                    t_arr.append(t)
                    Lx_arr.append(Lx_evo(t=t, track_dict=track_dict))

                    # calculate new envelope mass fraction:
                    M_env = M - M_core
                    f_env = (M_env/M)*100 # in %

                    # calculate new radius with new planet mass/envelope mass 
                    # fraction & one time step later
                    R = plmoLoFo14.calculate_planet_radius(M_core, f_env, t,
                                                       flux_at_planet_earth(
                                                           planet_object.Lbol,
                                                           planet_object.distance),
                                                       planet_object.metallicity)
                    R_arr.append(R)
                    i = i+1 # update step to i+1

                else:
                    sys.exit('sth went wrong of you see this!')
            break

        if (envelope_left == False):
            # planet has evaporated, so return last planet params for bare core
            return np.array(t_arr), np.array(M_arr),\
                   np.array(R_arr), np.array(Lx_arr)

    #print("Done!")
    return np.array(t_arr), np.array(M_arr), np.array(R_arr), np.array(Lx_arr)            


def mass_planet_RK4_forward_LO14_PAPER(epsilon, K_on, beta_on, planet_object, initial_step_size, t_final, track_dict):
    """USED: 4th order Runge-Kutta as numerical integration  method 
    Integrate from the current time (t_start (where planet has R0 and M0) into the future
    taking into account photoevaporative mass loss. 

    Input:
    ----------
    epsilon: evaporation efficiency
    K_on: set use of K parameter on or off 
    beta_on: set use of beta parameter on or off
    planet_object: object of planet class which contains also stellar parameters 
                   and info about stellar evo track
    step_size: initial step_size, fixed 
    t_final: final time of simulation
    track_dict: dictionary with Lx evolutionary track parameters
    
    Output:
    ----------
    t_arr, M_arr, R_arr, Lx_arr: array of time, mass, radius and Lx values from t_start to t_final
    """ 
    
    M_earth = const.M_earth.cgs.value
    R_earth = const.R_earth.cgs.value
    Myr_to_sec = 1e6*365*86400
    
    # initialize the starting values for Lxuv(t_start), mass, density, beta, K
    Lx0 = Lx_evo(t=track_dict["t_start"], track_dict=track_dict)
    Lxuv0 = L_xuv_all(Lx0) # use Sanz-Forcada2010 scaling law to get total XUV luminosity
    Fxuv0 = flux_at_planet(Lxuv0, planet_object.distance) # get flux at orbital separation a_p
    
    # initial planet parameters at t_start
    f_env_0 = f_env = planet_object.fenv 
    R0 = R = planet_object.radius # should match observed radius - determined by M_core and f_env
    M0 = M = planet_object.mass
    rho0 = rho = plmoLoFo14.density_planet(M0, R0)  # initial mean density
    M_env0 = M_env = M0 - planet_object.core_mass # initial envelope mass
    M_core = planet_object.core_mass
    
    # specify beta0 and K0
    if beta_on == "yes": # use the beta estimate from Salz et al. 2016
        beta = beta0 = bk.beta_fct(M0, Fxuv0, R0)
    elif beta_on == "no":
        beta = beta0 = 1.
    if K_on == "yes": # use K-correction estimation from Erkaev et al. 2007
        K = K0 = bk.K_fct(planet_object.distance, M0, planet_object.mass_star, R0)
    elif K_on == "no":
        K = K0 = 1.
    
    t_start = track_dict["t_start"]
    t_max = t_final
    step_size = initial_step_size
    
    # create time array for integration (with user-specified step size)
    number = math.ceil((t_max-t_start)/step_size)
    times, step_size2 = np.linspace(t_start, t_max, number, endpoint=True, retstep=True)
    #print('stepsize=', step_size2)
    
    # here I make lists of all the values I would like to track & output in the end:
    M_arr = [0]*len(times)
    M_arr[0] = M = M0 # inital value for Mdot at t_curr
    R_arr = [0]*len(times)
    R_arr[0] = R = R0
    t_arr = [0]*len(times)
    t_arr[0] = t_start  # inital value for t
    Lx_arr = [0]*len(times)
    #Lx_arr[0] = Lx0
    
    # CRITERION when to stop the mass-loss
    # for the Lopez planets I have a specified core mass and thus a fixed core radius (bare rocky core)
    R_core = planet_object.core_radius # stop when this radius is reached!
    
    dt = step_size2
    #for i in tqdm(range(1, len(times))):
    for i in range(0, len(times)-1):   
        #print(t_arr[i])
        ################
        # this is just for me to return the Lx(t) evolution to check if it is correct (not required since the Lx(t)
        # calculation is embedded in the mass_loss_rate_fancy function)
        Lx_i = Lx_evo(t=t_arr[i], track_dict=track_dict)
        
        ###############################################################################################################
        Mdot1 = mass_loss_rate_forward_LO14(times[i], epsilon, K_on, beta_on, planet_object, f_env, track_dict)
        k1 = (dt*Myr_to_sec * Mdot1)/M_earth # mass lost in one timestep in earth masses
        M_05k1 = M + 0.5*k1     
        M_env_05k1 = M_05k1 - M_core
        f_env_05k1 = (M_env_05k1/M_05k1)*100 # new mass fraction  
        
        Mdot2 = mass_loss_rate_forward_LO14(times[i]+0.5*dt, epsilon, K_on, beta_on, planet_object, f_env_05k1, track_dict)
        k2 = (dt*Myr_to_sec * Mdot2)/M_earth
        M_05k2 = M + 0.5*k2
        M_env_05k2 = M_05k2 - M_core
        f_env_05k2 = (M_env_05k2/M_05k2)*100 # new mass fraction
        
        Mdot3 = mass_loss_rate_forward_LO14(times[i]+0.5*dt, epsilon, K_on, beta_on, planet_object, f_env_05k2, track_dict) 
        k3 = (dt*Myr_to_sec * Mdot3)/M_earth
        M_k3 = M + k3
        M_env_k3 = M_k3 - M_core
        f_env_k3 = (M_env_k3/M_k3)*100 # new mass fraction
        
        Mdot4 = mass_loss_rate_forward_LO14(times[i]+dt, epsilon, K_on, beta_on, planet_object, f_env_k3, track_dict) 
        k4 = (dt*Myr_to_sec * Mdot4)/M_earth
        
        ###############################################################################################################
        M_lost = (k1 + 2*k2 + 2*k3 + k4)/6. # after time-step dt 
        ###############################################################################################################
        
        # check if planet with new mass does still have some atmosphere:
        if ((M + M_lost) - M_core) >= 0:
            # then planet still has some atmosphere left -> continue
            
            M_arr[i+1] = M = M + M_lost # new planet mass (assume envelope mass is lost)
            M_env = M - M_core # new envelope mass
            #M_arr[i] = M = M + (k1 + 2*k2 + 2*k3 + k4)/6. # M_i_plus_1 - update Mass value
            t_arr[i+1] = t = t_arr[i] + dt #t_start + i*dt # t_i_plus_1 - update time value
            # new envelope mass fraction:
            f_env = (M_env/M)*100 # in %
            # calculate new radius with new planet mass/envelope mass fraction & one time step later          
            R_arr[i+1] = R = plmoLoFo14.calculate_planet_radius(M_core, f_env, t, flux_at_planet_earth(planet_object.Lbol,
                                                            planet_object.distance), planet_object.metallicity)
            
        else:
            # all atmosphere is gone -> terminate
            print("Atmosphere has evaportated! Only bare rocky core left! STOP this madness!")

            # if the stop criterium is reached, I add the core mass & core radius to the array at t_i+1
            t_arr = np.trim_zeros(t_arr)
            print(t_arr[-1]+dt)
            #print(t_arr[i]+dt)
            t_arr = np.append(np.array(t_arr), t_arr[-1]+dt)
            M_arr = np.trim_zeros(M_arr)
            #M_arr = np.append(np.array([mass.value for mass in M_arr]), M_core.value)*u.g
            M_arr = np.append(np.array(M_arr), M_core)
            R_arr = np.trim_zeros(R_arr)
            #R_arr = np.append(np.array([radius.value for radius in R_arr]), R_core.value)*u.cm 
            R_arr = np.append(np.array(R_arr), R_core)
            Lx_arr = np.trim_zeros(Lx_arr)
            Lx_arr = np.array(Lx_arr.append(Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))) 
            return t_arr, M_arr, R_arr, Lx_arr

    Lx_arr[i+1] = Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict)
    Lx_arr = np.array(Lx_arr)
    t_arr = np.array(t_arr)
    M_arr = np.array(M_arr)
    R_arr = np.array(R_arr)
    print("Done!")
    return t_arr, M_arr, R_arr, Lx_arr


def mass_planet_RK4_forward_Ot14(epsilon, K_on, beta_on, planet_object, initial_step_size, t_final, track_dict):
    """USED: 4th order Runge-Kutta as numerical integration  method 
    Integrate from the current time (t_start (where planet has R0 and M0) into the future
    taking into account photoevaporative mass loss. 

    Input:
    ----------
    epsilon: evaporation efficiency
    K_on: set use of K parameter on or off 
    beta_on: set use of beta parameter on or off
    planet_object: object of planet class which contains also stellar parameters 
                   and info about stellar evo track
    step_size: initial step_size, variable
    [NOTE: the implementation of a variable step size is somewhat preliminary. The step size is adjusted 
    (made smaller or bigger depending how fast or slow the mass/radius changes) until the final time step 
    greater than t_final. This means that if the step size in the end is e.g. 10 Myr, and the integration 
    is at 4999 Myr, then last time entry will be 4999+10 -> 5009 Myr.]
    t_final: final time of simulation
    track_dict: dictionary with Lx evolutionary track parameters
    
    Output:
    ----------
    t_arr, M_arr, R_arr, Lx_arr: array of time, mass, radius and Lx values from t_start to t_final
    """ 
    
    M_earth = const.M_earth.cgs.value
    R_earth = const.R_earth.cgs.value
    Myr_to_sec = 1e6*365*86400
    
    # initialize the starting values for Lxuv(t_start), mass, density, beta, K
    Lx0 = Lx_evo(t=track_dict["t_start"], track_dict=track_dict)
    Lxuv0 = L_xuv_all(Lx0) # use scaling law to get total XUV luminosity
    Fxuv0 = flux_at_planet(Lxuv0, planet_object.distance) # get flux at orbital separation a_p
    
    # "make" initial planet at t_start
    R0 = R = planet_object.radius # should match observed radius
    M0 = M = planet_object.mass # planetary mass estimate based on mass-radius relation
    rho0 = rho = plmoOt20.density_planet(M0, R0)  # initial approx. density
    
    # specify beta0 and K0
    if beta_on == "yes": # use the beta estimate from Salz et al. 2016
        beta = beta0 = bk.beta_fct(M0, Fxuv0, R0)
    elif beta_on == "no":
        beta = beta0 = 1.
    if K_on == "yes": # use K-correction estimation from Erkaev et al. 2007
        K = K0 = bk.K_fct(planet_object.distance, M0, planet_object.mass_star, R0)
    elif K_on == "no":
        K = K0 = 1.

    M_arr = []
    M_arr.append(M0)
    R_arr = []
    R_arr.append(R0) 
    t_arr = []
    t0 = t = track_dict["t_start"]
    t_arr.append(t0) 
    Lx_arr = []
    Lx_arr.append(Lx0)
    #print(M0, R0, t0)
    
    # CRITERION when to stop the mass-loss
    R_core = 2.15 # stop when this radius is reached! (this is the minimum radius for which the volatile regime is valid)
    M_core = plmoOt20.calculate_mass_planet_Ot19(R_core)
    
    dt = initial_step_size
    i = 1 # counter
    while t <= t_final:
        #print(i, ' - ', t, "- dt = ", dt)
        ################
        # this is just for me to return the Lx(t) evolution to check if it is correct (not required since the Lx(t)
        # calculation is embedded in the mass_loss_rate_fancy function)
        Lx_i = Lx_evo(t=t, track_dict=track_dict)
        ################

        # for first step use the parameters initialized above (beta=beta0, K=K0, M=M0)
        ###############################################################################################################
        Mdot1 = mass_loss_rate_forward_Ot20(t, epsilon, K_on, beta_on, planet_object, R, track_dict) # mass M, radius R
        k1 = (dt*Myr_to_sec * Mdot1)/M_earth # mass lost in one timestep in earth masses
        M_05k1 = M + 0.5*k1     
        R_05k1 = plmoOt20.calculate_radius_planet_Ot19(M_05k1)
        
        Mdot2 = mass_loss_rate_forward_Ot20(t+0.5*dt, epsilon, K_on, beta_on, planet_object, R_05k1, track_dict)
        k2 = (dt*Myr_to_sec * Mdot2)/M_earth
        M_05k2 = M + 0.5*k2
        R_05k2 = plmoOt20.calculate_radius_planet_Ot19(M_05k2)
        
        Mdot3 = mass_loss_rate_forward_Ot20(t+0.5*dt, epsilon, K_on, beta_on, planet_object, R_05k2, track_dict) 
        k3 = (dt*Myr_to_sec * Mdot3)/M_earth
        M_05k3 = M + 0.5*k3
        R_05k3 = plmoOt20.calculate_radius_planet_Ot19(M_05k3)
        
        Mdot4 = mass_loss_rate_forward_Ot20(t+dt, epsilon, K_on, beta_on, planet_object, R_05k3, track_dict) 
        k4 = (dt*Myr_to_sec * Mdot4)/M_earth

        ###############################################################################
        M_lost = (k1 + 2*k2 + 2*k3 + k4)/6. # after time-step dt
        ###############################################################################
        
        # this part is new compared to the one used in the PAPER
        ###############################################################################
        # if radius change is too drastic, decrease step size
        M_new = M+M_lost
        R_new = plmoOt20.calculate_radius_planet_Ot19(M_new)
  
        # adjust step size, if radius change is too drastic or too little
        if (abs((R-R_new)/R)*100 >= 1.) & (t < track_dict["t_curr"]) & (dt > 1e-2):#10**(-1)): # smaller than 1%
            dt = dt/10.
            #print("radius change: ", abs((R-R_new)/R)*100)
            #print('make step size smaller: ', dt)
            # don't write anything to file, do iteration again with new step size
        elif (abs((R-R_new)/R)*100 < (0.01)) & (t < track_dict["t_curr"]) & (dt < 10.):
            dt = dt*10.
            #print('make step size bigger: ', dt)
        
        elif (abs((R-R_new)/R)*100 >= 1) & (t >= track_dict["t_curr"]) & (dt > 1e-1): # smaller than 1%
            dt = dt/10.
            #print('make step size smaller: ', dt)
        
        elif (abs((R-R_new)/R)*100 < (0.01)) & (t >= track_dict["t_curr"]) & (dt < 10.):
            dt = dt*10
            #print('make step size bigger: ', dt)
        
        else:#
            if ((M + M_lost) - M_core) >= 0:
                # then planet still has some atmosphere left -> continue
                M = M + M_lost # new planet mass (M_lost is negative)
                M_arr.append(M)
                t = t_arr[-1] + dt #t_start + i*dt # t_i_plus_1 - update time value
                t_arr.append(t) # new time t
                Lx_arr.append(Lx_evo(t=t, track_dict=track_dict)) # Lx at new time t

                # calculate new radius with new planet mass & one time step later
                R = plmoOt20.calculate_radius_planet_Ot19(M) 
                R_arr.append(R)
                i = i+1 # update step to i+1
                
            else:
                # all atmosphere is gone -> terminate
                print("Atmosphere has evaportated! Only bare rocky core left! STOP this madness!")

                # if the stop criterium is reached, I add the core mass & core radius to the array at t_i+1
                print("t = ", t_arr[-1]+dt)
                t_arr.append(t_arr[-1]+dt)
                M_arr.append(M_core)
                R_arr.append(R_core)
                Lx_arr.append(Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))            
                return np.array(t_arr), np.array(M_arr), np.array(R_arr), np.array(Lx_arr)
                              
    print("Done!")
    return np.array(t_arr), np.array(M_arr), np.array(R_arr), np.array(Lx_arr)


def mass_planet_RK4_forward_Ot14_PAPER(epsilon, K_on, beta_on, planet_object, initial_step_size, t_final, track_dict):
    """USED: 4th order Runge-Kutta as numerical integration  method 
    Integrate from the current time (t_start (where planet has R0 and M0) into the future
    taking into account photoevaporative mass loss. 

    Input:
    ----------
    epsilon: evaporation efficiency
    K_on: set use of K parameter on or off 
    beta_on: set use of beta parameter on or off
    planet_object: object of planet class which contains also stellar parameters 
                   and info about stellar evo track
    step_size: initial step_size, fixed
    t_final: final time of simulation
    track_dict: dictionary with Lx evolutionary track parameters
    
    Output:
    ----------
    t_arr, M_arr, R_arr, Lx_arr: array of time, mass, radius and Lx values from t_start to t_final
    """ 
    
    M_earth = const.M_earth.cgs.value
    R_earth = const.R_earth.cgs.value
    Myr_to_sec = 1e6*365*86400
    
    # initialize the starting values for Lxuv(t_start), mass, density, beta, K
    Lx0 = Lx_evo(t=track_dict["t_start"], track_dict=track_dict)
    Lxuv0 = L_xuv_all(Lx0) # use scaling law to get total XUV luminosity
    Fxuv0 = flux_at_planet(Lxuv0, planet_object.distance) # get flux at orbital separation a_p
    
    # "make" initial planet at t_start
    R0 = R = planet_object.radius # should match observed radius
    M0 = M = planet_object.mass # planetary mass estimate based on mass-radius relation
    rho0 = rho = plmoOt20.density_planet(M0, R0)  # initial approx. density
    
    # specify beta0 and K0
    if beta_on == "yes": # use the beta estimate from Salz et al. 2016
        beta = beta0 = bk.beta_fct(M0, Fxuv0, R0)
    elif beta_on == "no":
        beta = beta0 = 1.
    if K_on == "yes": # use K-correction estimation from Erkaev et al. 2007
        K = K0 = bk.K_fct(planet_object.distance, M0, planet_object.mass_star, R0)
    elif K_on == "no":
        K = K0 = 1.

    # create time array for integration (with user-specified step size)
    t_start, t_max = track_dict["t_start"], t_final
    step_size = initial_step_size
    number = math.ceil((t_max-t_start)/step_size)
    times, step_size2 = np.linspace(t_start, t_max, number, endpoint=True, retstep=True)
    #print('stepsize=', step_size2)
    
    # make lists of all the values wwe want to track & output in the end:
    M_arr = [0]*len(times)
    M_arr[0] = M = M0 # inital value for Mdot at t_curr
    R_arr = [0]*len(times)
    R_arr[0] = R = R0
    t_arr = [0]*len(times)
    t_arr[0] = t_start  # inital value for t
    Lx_arr = [0]*len(times)
    #Lx_arr[0] = Lx0
    
    # CRITERION when to stop the mass-loss
    R_core = 2.15 # stop when this radius is reached! (this is the minimum radius 
                  # for which the volatile regime is valid)
    M_core = plmoOt20.calculate_mass_planet_Ot19(R_core)
    
    dt = step_size2
    for i in range(0, len(times)-1):   
        # this is just for me to return the Lx(t) evolution to check if it is correct 
        # (not required since the Lx(t) calculation is embedded in 
        # the mass_loss_rate_fancy function)
        Lx_i = Lx_evo(t=t_arr[i], track_dict=track_dict)
        
        ###############################################################################################################
        Mdot1 = mass_loss_rate_forward_Ot20(times[i], epsilon, K_on, beta_on, planet_object, R, track_dict) # mass M, radius R
        k1 = (dt*Myr_to_sec * Mdot1)/M_earth # mass lost in one timestep in earth masses
        M_05k1 = M + 0.5*k1     
        R_05k1 = plmoOt20.calculate_radius_planet_Ot19(M_05k1)
        
        Mdot2 = mass_loss_rate_forward_Ot20(times[i]+0.5*dt, epsilon, K_on, beta_on, planet_object, R_05k1, track_dict)
        k2 = (dt*Myr_to_sec * Mdot2)/M_earth
        M_05k2 = M + 0.5*k2
        R_05k2 = plmoOt20.calculate_radius_planet_Ot19(M_05k2)
        
        Mdot3 = mass_loss_rate_forward_Ot20(times[i]+0.5*dt, epsilon, K_on, beta_on, planet_object, R_05k2, track_dict) 
        k3 = (dt*Myr_to_sec * Mdot3)/M_earth
        M_05k3 = M + 0.5*k3
        R_05k3 = plmoOt20.calculate_radius_planet_Ot19(M_05k3)
        
        Mdot4 = mass_loss_rate_forward_Ot20(times[i]+dt, epsilon, K_on, beta_on, planet_object, R_05k3, track_dict) 
        k4 = (dt*Myr_to_sec * Mdot4)/M_earth

        ###############################################################################################################
        M_lost = (k1 + 2*k2 + 2*k3 + k4)/6. # mass lost after time-step dt
        ###############################################################################################################

        # check if planet with new mass does still have some atmosphere
        if ((M + M_lost) - M_core) >= 0:
            # then planet still has some atmosphere left -> continue
            
            M_arr[i+1] = M = M + M_lost # new planet mass (M_lost is negative)
            t_arr[i+1] = t = t_arr[i] + dt #t_start + i*dt # t_i_plus_1 - update time value
            # calculate new radius with new planet mass
            R_arr[i+1] = R = plmoOt20.calculate_radius_planet_Ot19(M) 
            
        else:
            # all atmosphere is gone (based on criterion set at the top)-> terminate
            print("Atmosphere has evaportated! Only bare rocky core left! STOP this madness!")

            # if the stop criterium is reached, I add the core mass & core radius to the array at t_i+1
            t_arr = np.trim_zeros(t_arr)
            print(t_arr[-1]+dt)
            t_arr = np.append(np.array(t_arr), t_arr[-1]+dt)
            M_arr = np.trim_zeros(M_arr)
            M_arr = np.append(np.array(M_arr), M_core)
            R_arr = np.trim_zeros(R_arr)
            R_arr = np.append(np.array(R_arr), R_core)
            Lx_arr = np.trim_zeros(Lx_arr)
            Lx_arr = np.array(Lx_arr.append(Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))) 
            return t_arr, M_arr, R_arr, Lx_arr
    
    # if planet survives, output the final arrays
    Lx_arr[i+1] = Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict)
    Lx_arr = np.array(Lx_arr)
    t_arr = np.array(t_arr)
    M_arr = np.array(M_arr)
    R_arr = np.array(R_arr)
    print("Done!")
    return t_arr, M_arr, R_arr, Lx_arr
