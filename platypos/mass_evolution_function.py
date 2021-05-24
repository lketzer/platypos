import sys
import numpy as np
import pandas as pd
import math
import astropy.units as u
from astropy import constants as const

import platypos.planet_models_LoFo14 as plmoLoFo14
import platypos.planet_models_ChRo16 as plmoChRo16
import platypos.planet_model_Ot20 as plmoOt20
import platypos.beta_K_functions as bk
from platypos.mass_loss_rate_function import mass_loss_rate

from platypos.lx_evo_and_flux import lx_evo, l_xuv_all
from platypos.lx_evo_and_flux import flux_at_planet


def mass_evo_RK4_forward(planet_object,
                         track_dict,
                         mass_loss_calc="Elim",
                         epsilon=None,
                         K_on="yes", beta_settings={'beta_calc': 'off'},
                         initial_step_size=0.1,
                         t_final=5.0*1e9,
                         relation_EUV="Linsky",
                         fenv_sample_cut=False):
    """USED: 4th order Runge-Kutta as numerical integration  method 
    Integrate from the current time (t_start (where planet has R0 and M0) 
    into the future taking into account photoevaporative mass loss. 

    Parameters:
    -----------
    planet_object (class obj): object of one of the three planet classes
                               (Planet_LoFo14, Planet_ChRo16, Planet_Ot20),
                               which contains planetray & stellar parameters;
                               e.g. core mass needed to calculate the
                               current mass and radius of the planet)                 
    track_dict (dict): dictionary with evolutionary track parameters
    
    mass_loss_calc (str): ADD MORE DETAILS!!
                          "Elim", "RRlim", "Elim_and_RRlim", "HBA"

    epsilon (float): evaporation efficiency
    K_on (float): set use of K parameter on or off ("yes" or "no");
                  default is "yes"
                   
    beta_settings (dict): 
            Depending on the estimation procedure, the dictionary
            has 2 or 3 params.
            1) beta_calc (str): "Salz16" or "Lopez17" or "off"
                        a) approximation from a study by Salz et al. (2016)
                            NOTE from paper: "The atmospheric expansion can be
                            neglected for massive hot Jupiters, but in the range of
                            superEarth-sized planets the expansion causes mass-loss
                            rates that are higher by a factor of four."
                        b) approximation from MurrayClay (2009) or Lopez (2017)
                        c) beta = 1
            2) RL_cut (bool): if R_XUV > R_RL, set R_XUV=R_RL
            3) beta_cutoff (bool): additional parameter if beta_calc="Salz16";
                        - IF cutoff == True, beta is kept constant at the lower
                        boundary value for planets with gravities lower than
                        the Salz sample
                        - IF cutoff == False, the user agrees to extrapolate
                        this relation beyond the Salz-sample limits.                     

    
    step_size (float): initial step_size, variable
    t_final (float): final time of simulation (in Myr!)
    relation_EUV (str): "Linsky" OR "SanzForcada" -> estimate the EUV luminosity
                    using the X-ray-EUV scaling relations given by
                    Linsky et al. (2013, 2015) OR Sanz-Forcada et al. (2011)

    
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
    R_arr (array): radius array with radius evolution over time (from
                   thermal contraction and photoevaporative mass-loss)
    Lx_arr (array): array to trace the X-ray luminosity (mainly for
                    consistency checks)
    """ 
    # define some constants
    M_EARTH = 5.972364730419773e+27 #const.M_earth.cgs.value
    R_EARTH = 637810000.0 #const.R_earth.cgs.value
    Myr_to_sec = 1e6*365*86400
    
    # initialize the starting values for the X-ray luminosity at t_start,
    # the starting planet parameters, as well as beta & K (at t_start);
    # convert X-ray to XUV lum. and calculate planet's high energy 
    # incident flux
    Lx0 = lx_evo(t=track_dict["t_start"], track_dict=track_dict)
    Lxuv0 = l_xuv_all(Lx0, relation_EUV, planet_object.mass_star)
    Fxuv0 = flux_at_planet(Lxuv0, planet_object.distance)

    R0 = R = planet_object.radius
    M0 = M = planet_object.mass
    if ((planet_object.planet_type == "LoFo14") or
               (planet_object.planet_type == "ChRo16")):
        f_env_0 = f_env = planet_object.fenv
        M_env0 = M_env = M0 - planet_object.core_mass
        M_core = planet_object.core_mass
        R_core = planet_object.core_radius
    
    elif planet_object.planet_type == "Ot20":
        R_core = 2.15
        M_core = plmoOt20.calculate_mass_planet_Ot20(R_core)
        M_env0 = M_env = M0 - M_core
     
    # CRITERION for when the planet has lost all atmosphere
    # for the LoFo14 planets the core mass and thus the core radius is fixed.
    # So when the planet mass gets smaller or equal to the core mass, we
    # assume only the bare rocky core is left.
    # For Ot20 planets, terminate if R < 2.15 R_earth (hardcoded radius) is
    # reached! (aka the minimum radius for which the volatile regime is valid)
    
    #rho0 = rho = plmoLoFo14.density_planet(M0, R0)
    rho0 = rho = ((M0 * M_EARTH) / (4./3 * np.pi * (R0 * R_EARTH)**3))

    # since the step size is adaptive, I use lists to keep track of
    # the time, mass, radius and Lx evolution
    M_arr = []
    M_arr.append(M0)
    R_arr = []
    R_arr.append(R0) 
    t_arr = []
    t0 = t = track_dict["t_start"]
    t_arr.append(t0) 
    Lx_arr = []
    Lx_arr.append(Lx0)
    
    # I want to save the info about the calc. method
    Mdot_info = []

    dt = initial_step_size
    # NOTE: minimum and maximum step size are HARDCODED for now (see further 
    # down in code for more details)
    min_step_size, max_step_size = 1e-2,  10.

    i = 1  # counter to track how many traced RK iterations have been performed
    j = 1  # counter to track how many RK iterations have been attempted.
    envelope_left = True  # variable to flag a planet if envelope is gone
    close_to_evaporation = False  # variable to flag if planet is close to
                                  # complete atmospheric removal

    # make list with all step sizes, even those which resulted in too drastic 
    # radius changes -> then I can check if the code gets stuck in an infinite
    # loop between make_bigger, make_smaller, make_bigger, etc..
    step_size_list = []
    number_of_convergence_checks = 0
    
    while t <= t_final:
        #print(i, j, " t= ", t, dt)

        # This step (Lx(t) calculation) is just for me to trace Lx and check 
        # if it is correct. It is NOT required since the Lx(t) calculation is 
        # embedded in the mass_loss_rate_fancy function)
        Lx_i = lx_evo(t=t, track_dict=track_dict)

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
        # 2) at the end of planet evolution when the planet mass gets very
        # close to the core mass, at some point the mass lost is larger than
        # the renmaining atmosphere mass (either at the end of a coplete RK
        # iteration, or this already happends in one of the RK substeps, in
        # which case the mass lost after the complete RK step evaluates to nan
        # and no new planet radius can be calculated). In both cases the planet
        # is assumed to be fully evaporated at t_i + dt.

        # add the current step size
        step_size_list.append(dt)
        # take the last 20 entries in step_size_array and check for 
        # constant back-and-forth between two step sizes
        # e.g. [0.1, 1.0, 0.1, 1.0, ...]
        # check if difference is the same, but values in array are not all 
        # the same; also need to check if every 2nd element is the same, 
        # same goes for 2(n+1)th
        step_size_array = np.array(step_size_list)
        step_size_difference = abs(np.diff(step_size_array[-20:]))

        if (len(step_size_array) >= 20): # check only after a 20 iterations
            step_size_array = step_size_array[-20:]
            if (np.all(step_size_difference == step_size_difference[0]) and 
                    ~np.all(step_size_array == step_size_array[0]) and 
                    np.all(step_size_array[::2] == step_size_array[0]) and 
                    np.all(step_size_array[1::2] == step_size_array[1])):
                #print("no convergence, set min. step size.")
                number_of_convergence_checks += 1
                # if no convergence, switch to minumum step size
                dt = min_step_size
        if number_of_convergence_checks > 5:
            #print("end calculation with min. step size")
            close_to_evaporation = True
            dt = min_step_size
        
        # else, all is good, continue with current step size
        while (envelope_left == True):
            # go through RK iterations as long as there is envelope left
            
            # apply Runge Kutta 4th order to find next value of M_dot
            # NOTE: the mass lost in one timestep is in Earth masses
            if ((planet_object.planet_type == "LoFo14") or
               (planet_object.planet_type == "ChRo16")):
                if (fenv_sample_cut == True) & (f_env < 1e-2):\
                    f_env = 1e-2
                Mdot1, info1 = mass_loss_rate(t_=t,
                                       track_dict=track_dict,
                                       planet=planet_object,
                                       mass_loss_calc=mass_loss_calc,
                                       epsilon=epsilon,
                                       K_on=K_on,
                                       beta_settings=beta_settings,
                                       f_env=f_env,
                                       relation_EUV=relation_EUV)
            elif planet_object.planet_type == "Ot20":
                Mdot1, info1 = mass_loss_rate(t_=t,
                                       track_dict=track_dict,
                                       planet=planet_object,
                                       mass_loss_calc=mass_loss_calc,
                                       epsilon=epsilon,
                                       K_on=K_on,
                                       beta_settings=beta_settings,
                                       radius_at_t_=R,
                                       relation_EUV=relation_EUV)
            
            k1 = (dt * Myr_to_sec * Mdot1) / M_EARTH 
            M_k1 = M + 0.5 * k1 # mass after 1st RK step
            
            if ((planet_object.planet_type == "LoFo14") or
               (planet_object.planet_type == "ChRo16")):
                M_env_k1 = M_k1 - M_core
                f_env_k1 = (M_env_k1 / M_k1) * 100 # new envelope mass frac.
            elif planet_object.planet_type == "Ot20":
                R_k1 = plmoOt20.calculate_radius_planet_Ot20(M_k1)
                
            if (i == 1) and (j == 1) and (M_k1 < M_core):
                # then I am still in the first RK iteration, and the initial
                # step size was likely too large -> set step size to minumum
                dt = min_step_size
                j += 1
                break
            
            if ((planet_object.planet_type == "LoFo14") or
               (planet_object.planet_type == "ChRo16")):
                if (fenv_sample_cut == True) & (f_env_k1 < 1e-2):\
                    f_env_k1 = 1e-2
                Mdot2, info2 = mass_loss_rate(t_=t + 0.5*dt,
                                       track_dict=track_dict,
                                       planet=planet_object,
                                       mass_loss_calc=mass_loss_calc,
                                       epsilon=epsilon,
                                       K_on=K_on,
                                       beta_settings=beta_settings,
                                       f_env=f_env_k1,
                                       relation_EUV=relation_EUV)
            elif planet_object.planet_type == "Ot20":
                Mdot2, info2 = mass_loss_rate(t_=t + 0.5*dt,
                                       track_dict=track_dict,
                                       planet=planet_object,
                                       mass_loss_calc=mass_loss_calc,
                                       epsilon=epsilon,
                                       K_on=K_on,
                                       beta_settings=beta_settings,
                                       radius_at_t_=R_k1,
                                       relation_EUV=relation_EUV)
            
            k2 = (dt * Myr_to_sec * Mdot2) / M_EARTH
            M_k2 = M + 0.5 * k2
            
            if ((planet_object.planet_type == "LoFo14") or
               (planet_object.planet_type == "ChRo16")):
                M_env_k2 = M_k2 - M_core
                f_env_k2 = (M_env_k2 / M_k2) * 100    
            elif planet_object.planet_type == "Ot20":
                R_k2 = plmoOt20.calculate_radius_planet_Ot20(M_k2)
                
            if (i == 1) and (j == 1) and (M_k2 < M_core):
                # then I am still in the first RK iteration, and the initial
                # step size was likely too large -> set step size to minumum
                dt = min_step_size
                j += 1
                break

            if ((planet_object.planet_type == "LoFo14") or
               (planet_object.planet_type == "ChRo16")):
                if (fenv_sample_cut == True) & (f_env_k2 < 1e-2):\
                    f_env_k2 = 1e-2
                Mdot3, info3 = mass_loss_rate(t_=t + 0.5*dt,
                                       track_dict=track_dict,
                                       planet=planet_object,
                                       mass_loss_calc=mass_loss_calc,
                                       epsilon=epsilon,
                                       K_on=K_on,
                                       beta_settings=beta_settings,
                                       f_env=f_env_k2,
                                       relation_EUV=relation_EUV)
            elif planet_object.planet_type == "Ot20":
                Mdot3, info3 = mass_loss_rate(t_=t + 0.5*dt,
                                       track_dict=track_dict,
                                       planet=planet_object,
                                       mass_loss_calc=mass_loss_calc,
                                       epsilon=epsilon,
                                       K_on=K_on,
                                       beta_settings=beta_settings,
                                       radius_at_t_=R_k2,
                                       relation_EUV=relation_EUV)

            k3 = (dt * Myr_to_sec * Mdot3) / M_EARTH
            M_k3 = M + k3
            
            if ((planet_object.planet_type == "LoFo14") or
               (planet_object.planet_type == "ChRo16")):
                M_env_k3 = M_k3 - M_core
                f_env_k3 = (M_env_k3 / M_k3) * 100
            elif planet_object.planet_type == "Ot20":
                R_k3 = plmoOt20.calculate_radius_planet_Ot20(M_k3)

            if (i == 1) and (j == 1) and (M_k3 < M_core):
                # then I am still in the first RK iteration, and the initial
                # step size was likely too large -> set step size to minumum
                dt = min_step_size
                j += 1
                break
                
            if ((planet_object.planet_type == "LoFo14") or
               (planet_object.planet_type == "ChRo16")):
                if (fenv_sample_cut == True) & (f_env_k3 < 1e-2):\
                    f_env_k3 = 1e-2
                Mdot4, info4 = mass_loss_rate(t_=t + dt,
                                       track_dict=track_dict,
                                       planet=planet_object,
                                       mass_loss_calc=mass_loss_calc,
                                       epsilon=epsilon,
                                       K_on=K_on,
                                       beta_settings=beta_settings,
                                       f_env=f_env_k3,
                                       relation_EUV=relation_EUV)
            elif planet_object.planet_type == "Ot20":
                Mdot4, info4= mass_loss_rate(t_=t + dt,
                                       track_dict=track_dict,
                                       planet=planet_object,
                                       mass_loss_calc=mass_loss_calc,
                                       epsilon=epsilon,
                                       K_on=K_on,
                                       beta_settings=beta_settings,
                                       radius_at_t_=R_k3,
                                       relation_EUV=relation_EUV)

            k4 = (dt * Myr_to_sec * Mdot4) / M_EARTH
            # total mass lost after time-step dt
            
            #print(info1, info2, info3, info4)
            
            M_lost = (k1 + 2*k2 + 2*k3 + k4) / 6. 

            # update next value of the planet mass
            M_new = M + M_lost
            M_env_new = M_new - M_core
            #print(t, dt, M_new)
            
            # now it is time to check if atmosphere is gone or 
            # if planet is close to complete atmosphere removal
            # for Otegi planets, check if M_new < M_core
            if ((np.isnan(M_lost) == True) \
                    or (np.iscomplex(M_new) == True) or (M_new < M_core)) \
                    and (dt == min_step_size):
                # if M_lost = nan (i.e. planet evaporates in one of the RK
                # steps) OR the four RK steps finish and the new planet mass is
                # smaller or equal to the core mass, then the planet counts as
                # evaporated!
                # if this condition is satisfied and the step size is already
                # at a minimum, then we assume the current RK iteration would
                # remove all atmosphere and only the rocky core is left at
                # t_i+dt; this terminates the code and returns the final planet
                # properties

                #print("Atmosphere has evaportated! Only bare rocky core left!"\
                #       + " STOP this madness!")

                # since the the stop criterium is reached, we assume at t_i+1
                # the planet only consists of the bare rocky core with the
                # planet mass equal to the core mass and the planet radius
                # equal to the core radius
                t_arr.append(t_arr[-1]+dt)
                M_arr.append(M_core)
                R_arr.append(R_core)
                Lx_arr.append(lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))
                Mdot_info.append(info1)
                envelope_left = False  # set flag for complete env. removal
                j += 1
                break
                
            elif ((np.isnan(M_lost) == True) \
                    or (np.iscomplex(M_new) == True) or (M_new < M_core)) \
                    and (dt > min_step_size) \
                    and (close_to_evaporation == False):
                #print("close to evaporation")
                # planet close to evaporation, but since the step size is not
                # minimum yet, we set it to its minimum value and run the RK
                # iteration again (until above stopping condition is fulfilled)
                dt = min_step_size
                close_to_evaporation = True
                # this variable is for making sure the code does not run into
                # an infinite loop when the planet is close to evaporation.
                # Once this condition is set to True, the code continues with
                # a fixed min. step size and is no longer allowed to adjust it.
                j += 1
                break
            
            # this part is new compared to the one used in the PAPER (there we
            # used a fixed step size!)
            # if you're still in the while loop at this point, then calculate
            # new radius and check how drastic the radius change would be;
            # adjust the step size if too drastic or too little
            if planet_object.planet_type == "LoFo14":
                f_env_new = (M_env_new / M_new) * 100 # in %
                if (fenv_sample_cut == True) & (f_env_new < 1e-2):
                    R_new = plmoLoFo14.calculate_planet_radius(
                                                M_core, 1e-2, t,
                                                planet_object.flux,
                                                planet_object.metallicity
                                                )
                else:
                    R_new = plmoLoFo14.calculate_planet_radius(
                                                M_core, f_env_new, t,
                                                planet_object.flux,
                                                planet_object.metallicity
                                                )
            elif planet_object.planet_type == "ChRo16":
                f_env_new = (M_env_new / M_new) * 100 # in %
                if (fenv_sample_cut == True) & (f_env_new < 1e-2):
                    R_new = plmoChRo16.calculate_planet_radius(
                                                M_core, 1e-2,
                                                planet_object.flux, t,
                                                planet_object.core_comp
                                                )
                else:
                    R_new = plmoChRo16.calculate_planet_radius(
                                                M_core, f_env_new,
                                                planet_object.flux, t,
                                                planet_object.core_comp
                                                )
            elif planet_object.planet_type == "Ot20":
                R_new = plmoOt20.calculate_radius_planet_Ot20(M_new)
            
            #print("R_new, f_new: ", R_new, f_env_new)
            #print(abs((R-R_new)/R)*100)
            
            # only adjust step size if planet is not close to complete evaporat.
            if (close_to_evaporation == False):
                #print("radius check")
                # then do the check on how much the radius changes
                # R(t_i) compared to R(t_i+dt);
                # if radius change is larger than 0.5%, make step size smaller
                # by factor 10 OR if radius change is smaller than 0.02%, make
                # step size bigger by factor 10, but not bigger or smaller than
                # max. or min. step size!
                # if radius change too much/little, do not write anything to
                # file, instead do RK iteration again with new step size

                R_change = abs((R - R_new) / R) * 100 # radius change compared to
                                                # previous radius - in percent
            
            
            
            # only adjust step size if planet is not close to complete evaporat.
            if (close_to_evaporation == False):
                #print("radius check")
                # then do the check on how much the radius changes
                # R(t_i) compared to R(t_i+dt);
                # if radius change is larger than 0.5%, make step size smaller
                # by factor 10 OR if radius change is smaller than 0.02%, make
                # step size bigger by factor 10, but not bigger or smaller than
                # max. or min. step size!
                # if radius change too much/little, do not write anything to
                # file, instead do RK iteration again with new step size

                R_change = abs((R - R_new) / R) * 100 # radius change compared to
                                                # previous radius - in percent
                if (R_change > 0.5) \
                        and (t < track_dict["t_curr"]) \
                        and (dt > min_step_size):
                    dt = dt / 10.
                    j += 1
                    break

                #elif ((R_change < 1.) \
                #			and (R_change >=0.1)) \
                #			and (t < track_dict["t_curr"]) \
                #			and (dt > min_step_size):
                #    dt = dt / 10.
                #    break

                elif (R_change < (0.02)) \
                        and (t < track_dict["t_curr"]) \
                        and (dt < max_step_size):
                    dt = dt * 10.
                    j += 1
                    break    

                # NOTE: in principle I can adjust the code such that these 
                # hardcoded parameters are different for early planet evolution
                # where much more is happening typically, and late planet
                # evolution where almost no change is occurring anymore
                elif (R_change > 0.5) \
                        and (t >= track_dict["t_curr"]) \
                        and (dt > min_step_size):
                    dt = dt / 10.
                    j += 1
                    break

                elif (R_change < (0.02)) \
                        and (t >= track_dict["t_curr"]) \
                        and (dt < max_step_size):
                    dt = dt * 10
                    j += 1
                    break

                else: # if radius change is ok
                    # do sanity check: is new planet mass is still greater than
                    # the core mass? ->then there is still some atmosphere left
                    # in this case update params and go into next RK iteration
                    if ((M + M_lost) - M_core) > 0: 
                        M = M + M_lost  # new planet mass (M_lost is negative)
                        t = t_arr[-1] + dt  # updated time value t_i_plus_1
                        M_arr.append(M)
                        t_arr.append(t)
                        Lx_arr.append(lx_evo(t=t, track_dict=track_dict))
                        Mdot_info.append(info1)

                        # calculate new radius with new planet mass/envelope
                        # mass fraction & one time step later
                        if planet_object.planet_type == "LoFo14":
                            # calculate new envelope mass fraction:
                            M_env = M - M_core
                            f_env = (M_env / M) * 100 # in %
                            if (fenv_sample_cut == True) & (f_env < 1e-2):
                                R = plmoLoFo14.calculate_planet_radius(
                                                    M_core, 1e-2, t,
                                                    planet_object.flux,
                                                    planet_object.metallicity
                                                    )
                            else:
                                R = plmoLoFo14.calculate_planet_radius(
                                                    M_core, f_env, t,
                                                    planet_object.flux,
                                                    planet_object.metallicity
                                                    )
                        elif planet_object.planet_type == "ChRo16":
                            # calculate new envelope mass fraction:
                            M_env = M - M_core
                            f_env = (M_env / M) * 100 # in %
                            if (fenv_sample_cut == True) & (f_env < 1e-2):
                                R = plmoChRo16.calculate_planet_radius(
                                                M_core, 1e-2,
                                                planet_object.flux, t,
                                                planet_object.core_comp
                                                )
                            else:
                                R = plmoChRo16.calculate_planet_radius(
                                                M_core, f_env,
                                                planet_object.flux, t,
                                                planet_object.core_comp
                                                )
                        elif planet_object.planet_type == "Ot20":
                            R = plmoOt20.calculate_radius_planet_Ot20(M)

                        R_arr.append(R)
                        i += 1 # update step to i+1
                        j += 1

                    else:
                        # this should never happen
                        sys.exit('sth went wrong of you see this!')
                break
            
            elif (close_to_evaporation == True): 
                # if this condition is true, do not adjust step size
                # based on the radius change
                if ((M + M_lost) - M_core) > 0:
                    M = M + M_lost # new planet mass (M_lost is negative)
                    t = t_arr[-1] + dt #  updated time value t_i_plus_1
                    M_arr.append(M)
                    t_arr.append(t)
                    Lx_arr.append(lx_evo(t=t, track_dict=track_dict))
                    Mdot_info.append(info1)

                    # calculate new radius with new planet mass/envelope
                    # mass fraction & one time step later
                    if planet_object.planet_type == "LoFo14":
                        # calculate new envelope mass fraction:
                        M_env = M - M_core
                        f_env = (M_env / M) * 100 # in %
                        if (fenv_sample_cut == True) & (f_env < 1e-2):
                            R = plmoLoFo14.calculate_planet_radius(
                                                M_core, 1e-2, t,
                                                planet_object.flux,
                                                planet_object.metallicity
                                                )
                        else:
                            R = plmoLoFo14.calculate_planet_radius(
                                                M_core, f_env, t,
                                                planet_object.flux,
                                                planet_object.metallicity
                                                )

                    elif planet_object.planet_type == "ChRo16":
                        # calculate new envelope mass fraction:
                        M_env = M - M_core
                        f_env = (M_env / M) * 100 # in %
                        if (fenv_sample_cut == True) & (f_env < 1e-2):
                            R = plmoChRo16.calculate_planet_radius(
                                                M_core, 1e-2,
                                                planet_object.flux, t,
                                                planet_object.core_comp
                                                )
                        else:
                            R = plmoChRo16.calculate_planet_radius(
                                                M_core, f_env,
                                                planet_object.flux, t,
                                                planet_object.core_comp
                                                )
                            
                    elif planet_object.planet_type == "Ot20":
                        R = plmoOt20.calculate_radius_planet_Ot20(M)
                        
                    R_arr.append(R)
                    i += 1 # update step to i+1
                    j += 1

                else:
                    sys.exit('sth went wrong of you see this!')
            break

        if (envelope_left == False):
            # planet has evaporated, so return last planet params for bare core
            Mdot_info.append('End')
            return pd.DataFrame({"Time": t_arr, "Mass": M_arr,
                                 "Radius": R_arr, "Lx": Lx_arr,
                                 "Mdot_info": Mdot_info})

    #print("Done!")
    Mdot_info.append('End')
    return pd.DataFrame({"Time": t_arr, "Mass": M_arr,
                         "Radius": R_arr, "Lx": Lx_arr,
                         "Mdot_info": Mdot_info})




# def mass_evo_RK4_forward(planet_object,
#                          track_dict,
#                          mass_loss_calc="Elim",
#                          epsilon=None,
#                          K_on="yes", beta_on="yes",
#                          beta_cutoff=True,
#                          initial_step_size=0.1,
#                          t_final=5.0*1e9,
#                          relation_EUV="Linsky"):
#     """USED: 4th order Runge-Kutta as numerical integration  method 
#     Integrate from the current time (t_start (where planet has R0 and M0) 
#     into the future taking into account photoevaporative mass loss. 

#     Parameters:
#     -----------
#     planet_object (class obj): object of one of the three planet classes
#                                (Planet_LoFo14, Planet_ChRo16, Planet_Ot20),
#                                which contains planetray & stellar parameters;
#                                e.g. core mass needed to calculate the
#                                current mass and radius of the planet)                 
#     track_dict (dict): dictionary with evolutionary track parameters
    
#     mass_loss_calc (str): ADD MORE DETAILS!!
#                           "Elim", "RRlim", "Elim_and_RRlim", "HBA"

    
#     epsilon (float): evaporation efficiency
#     K_on (float): set use of K parameter on or off ("yes" or "no");
#                   default is "yes"
#     beta_on (float): set use of beta parameter on or off ("yes" or "no" or
#                      "yes_old");
#                      ("yes_old" is without Roche Lobe cutoff)
#                      default is "yes"
                     
#     For planets of type Planet_LoFo14 & Planet_ChRo16 if beta_on == 'yes':
#     - if cutoff == True, beta is kept constant at the lower boundary value
#     for planets with gravities lower than the Salz sample
#     - if cutoff == False, the user agrees to extrapolate this relation beyond
#     the Salz-sample limits.

    
#     step_size (float): initial step_size, variable
#     t_final (float): final time of simulation (in Myr!)
#     relation_EUV (str): "Linsky" OR "SanzForcada" -> estimate the EUV luminosity
#                     using the X-ray-EUV scaling relations given by
#                     Linsky et al. (2013, 2015) OR Sanz-Forcada et al. (2011)

    
#     [NOTE: the implementation of a variable step size is somewhat preliminary.
#     The step size is adjusted (made smaller or bigger depending how fast or
#     slow the mass/radius changes) until the final time step greater than
#     t_final. This means that if the step size in the end is e.g. 10 Myr, and
#     the integration is at 4999 Myr, then last time entry will be 4999+10 ->
#     5009 Myr.]

#     Returns:
#     --------
#     t_arr (array): time array to trace mass and radius evolution
#     M_arr (array): mass array with mass evolution over time (mass decrease)
#     R_arr (array): radius array with radius evolution over time (from
#                    thermal contraction and photoevaporative mass-loss)
#     Lx_arr (array): array to trace the X-ray luminosity (mainly for
#                     consistency checks)
#     """ 
#     # define some constants
#     M_EARTH = const.M_earth.cgs.value
#     Myr_to_sec = 1e6*365*86400
    
#     # initialize the starting values for the X-ray luminosity at t_start,
#     # the starting planet parameters, as well as beta & K (at t_start);
#     # convert X-ray to XUV lum. and calculate planet's high energy 
#     # incident flux
#     Lx0 = lx_evo(t=track_dict["t_start"], track_dict=track_dict)
#     Lxuv0 = l_xuv_all(Lx0, relation_EUV)
#     Fxuv0 = flux_at_planet(Lxuv0, planet_object.distance)

#     R0 = R = planet_object.radius
#     M0 = M = planet_object.mass
#     if ((planet_object.planet_type == "LoFo14") or
#                (planet_object.planet_type == "ChRo16")):
#         f_env_0 = f_env = planet_object.fenv
#         M_env0 = M_env = M0 - planet_object.core_mass
#         M_core = planet_object.core_mass
#         R_core = planet_object.core_radius
    
#     elif planet_object.planet_type == "Ot20":
#         R_core = 2.15
#         M_core = plmoOt20.calculate_mass_planet_Ot20(R_core)
#         M_env0 = M_env = M0 - M_core
     
#     # CRITERION for when the planet has lost all atmosphere
#     # for the LoFo14 planets the core mass and thus the core radius is fixed.
#     # So when the planet mass gets smaller or equal to the core mass, we
#     # assume only the bare rocky core is left.
#     # For Ot20 planets, terminate if R < 2.15 R_earth (hardcoded radius) is
#     # reached! (aka the minimum radius for which the volatile regime is valid)
    
#     rho0 = rho = plmoLoFo14.density_planet(M0, R0)

#     # since the step size is adaptive, I use lists to keep track of
#     # the time, mass, radius and Lx evolution
#     M_arr = []
#     M_arr.append(M0)
#     R_arr = []
#     R_arr.append(R0) 
#     t_arr = []
#     t0 = t = track_dict["t_start"]
#     t_arr.append(t0) 
#     Lx_arr = []
#     Lx_arr.append(Lx0)

#     dt = initial_step_size
#     # NOTE: minimum and maximum step size are HARDCODED for now (see further 
#     # down in code for more details)
#     min_step_size, max_step_size = 1e-2,  10.

#     i = 1  # counter to track how many traced RK iterations have been performed
#     j = 1  # counter to track how many RK iterations have been attempted.
#     envelope_left = True  # variable to flag a planet if envelope is gone
#     close_to_evaporation = False  # variable to flag if planet is close to
#                                   # complete atmospheric removal

#     # make list with all step sizes, even those which resulted in too drastic 
#     # radius changes -> then I can check if the code gets stuck in an infinite
#     # loop between make_bigger, make_smaller, make_bigger, etc..
#     step_size_list = []
    
#     while t <= t_final:
#         #print(i, j, " t= ", t, dt)

#         # This step (Lx(t) calculation) is just for me to trace Lx and check 
#         # if it is correct. It is NOT required since the Lx(t) calculation is 
#         # embedded in the mass_loss_rate_fancy function)
#         Lx_i = lx_evo(t=t, track_dict=track_dict)

#         # IMPORTANT points on the time step:
#         # When the initial time step is too large OR the planet mass becomes
#         # very close to the core mass (after several time steps), it can happen
#         # that one of the RK substeps leads to such a large mass lost that the
#         # new planet mass is smaller than the core mass.
#         # Distinguish between two cases:
#         # 1) initial time step is too large such that M_lost = nan after the
#         # first iteration (i.e. Rk substep mass < core mass)
#         # -> immediately switch to lowest possible step size and let code run
#         # from there (i.e. code will make step size bigger again if necessary)
#         # 2) at the end of planet evolution when the planet mass gets very
#         # close to the core mass, at some point the mass lost is larger than
#         # the renmaining atmosphere mass (either at the end of a coplete RK
#         # iteration, or this already happends in one of the RK substeps, in
#         # which case the mass lost after the complete RK step evaluates to nan
#         # and no new planet radius can be calculated). In both cases the planet
#         # is assumed to be fully evaporated at t_i + dt.

#         # add the current step size
#         step_size_list.append(dt)
#         # take the last 20 entries in step_size_array and check for 
#         # constant back-and-forth between two step sizes
#         # e.g. [0.1, 1.0, 0.1, 1.0, ...]
#         # check if difference is the same, but values in array are not all 
#         # the same; also need to check if every 2nd element is the same, 
#         # same goes for 2(n+1)th
#         step_size_array = np.array(step_size_list)
#         step_size_difference = abs(np.diff(step_size_array[-20:]))
#         if (len(step_size_array) >= 20): # check only after a 20 iterations
#             if (np.all(step_size_difference == step_size_difference[0]) and 
#                     ~np.all(step_size_array == step_size_array[0]) and 
#                     np.all(step_size_array[::2] == step_size_array[0]) and 
#                     np.all(step_size_array[1::2] == step_size_array[1])):
#                 print("no convergence, set min. step size.")
#                 # if no convergence, switch to minumum step size
#                 dt = min_step_size
        
#         # else, all is good, continue with current step size
#         while (envelope_left == True):
#             # go through RK iterations as long as there is envelope left
            
#             # apply Runge Kutta 4th order to find next value of M_dot
#             # NOTE: the mass lost in one timestep is in Earth masses
            
#             if ((planet_object.planet_type == "LoFo14") or
#                (planet_object.planet_type == "ChRo16")):
#                 Mdot1 = mass_loss_rate(t_=t,
#                                        track_dict=track_dict,
#                                        planet_object=planet_object,
#                                        mass_loss_calc=mass_loss_calc,
#                                        epsilon=epsilon,
#                                        K_on=K_on, beta_on=beta_on,
#                                        f_env=f_env,
#                                        beta_cutoff=beta_cutoff,
#                                        relation_EUV=relation_EUV)
#             elif planet_object.planet_type == "Ot20":
#                 Mdot1 = mass_loss_rate(t_=t,
#                                        track_dict=track_dict,
#                                        planet_object=planet_object,
#                                        mass_loss_calc=mass_loss_calc,
#                                        epsilon=epsilon,
#                                        K_on=K_on, beta_on=beta_on,
#                                        radius_at_t_=R,
#                                        beta_cutoff=beta_cutoff,
#                                        relation_EUV=relation_EUV)
            
#             k1 = (dt * Myr_to_sec * Mdot1) / M_EARTH 
#             M_k1 = M + 0.5 * k1 # mass after 1st RK step
            
#             if ((planet_object.planet_type == "LoFo14") or
#                (planet_object.planet_type == "ChRo16")):
#                 M_env_k1 = M_k1 - M_core
#                 f_env_k1 = (M_env_k1 / M_k1) * 100 # new envelope mass frac.
#             elif planet_object.planet_type == "Ot20":
#                 R_k1 = plmoOt20.calculate_radius_planet_Ot20(M_k1)
                
#             if (i == 1) and (j == 1) and (M_k1 < M_core):
#                 # then I am still in the first RK iteration, and the initial
#                 # step size was likely too large -> set step size to minumum
#                 dt = min_step_size
#                 j += 1
#                 break
            
#             if ((planet_object.planet_type == "LoFo14") or
#                (planet_object.planet_type == "ChRo16")):
#                 Mdot2 = mass_loss_rate(t_=t + 0.5*dt,
#                                        track_dict=track_dict,
#                                        planet_object=planet_object,
#                                        mass_loss_calc=mass_loss_calc,
#                                        epsilon=epsilon,
#                                        K_on=K_on, beta_on=beta_on,
#                                        f_env=f_env_k1,
#                                        beta_cutoff=beta_cutoff,
#                                        relation_EUV=relation_EUV)
#             elif planet_object.planet_type == "Ot20":
#                 Mdot2 = mass_loss_rate(t_=t + 0.5*dt,
#                                        track_dict=track_dict,
#                                        planet_object=planet_object,
#                                        mass_loss_calc=mass_loss_calc,
#                                        epsilon=epsilon,
#                                        K_on=K_on, beta_on=beta_on,
#                                        radius_at_t_=R_k1,
#                                        beta_cutoff=beta_cutoff,
#                                        relation_EUV=relation_EUV)
            
#             k2 = (dt * Myr_to_sec * Mdot2) / M_EARTH
#             M_k2 = M + 0.5 * k2
            
#             if ((planet_object.planet_type == "LoFo14") or
#                (planet_object.planet_type == "ChRo16")):
#                 M_env_k2 = M_k2 - M_core
#                 f_env_k2 = (M_env_k2 / M_k2) * 100    
#             elif planet_object.planet_type == "Ot20":
#                 R_k2 = plmoOt20.calculate_radius_planet_Ot20(M_k2)
                
#             if (i == 1) and (j == 1) and (M_k2 < M_core):
#                 # then I am still in the first RK iteration, and the initial
#                 # step size was likely too large -> set step size to minumum
#                 dt = min_step_size
#                 j += 1
#                 break

#             if ((planet_object.planet_type == "LoFo14") or
#                (planet_object.planet_type == "ChRo16")):
#                 Mdot3 = mass_loss_rate(t_=t + 0.5*dt,
#                                        track_dict=track_dict,
#                                        planet_object=planet_object,
#                                        mass_loss_calc=mass_loss_calc,
#                                        epsilon=epsilon,
#                                        K_on=K_on, beta_on=beta_on,
#                                        f_env=f_env_k2,
#                                        beta_cutoff=beta_cutoff,
#                                        relation_EUV=relation_EUV)
#             elif planet_object.planet_type == "Ot20":
#                 Mdot3 = mass_loss_rate(t_=t + 0.5*dt,
#                                        track_dict=track_dict,
#                                        planet_object=planet_object,
#                                        mass_loss_calc=mass_loss_calc,
#                                        epsilon=epsilon,
#                                        K_on=K_on, beta_on=beta_on,
#                                        radius_at_t_=R_k2,
#                                        beta_cutoff=beta_cutoff,
#                                        relation_EUV=relation_EUV)

#             k3 = (dt * Myr_to_sec * Mdot3) / M_EARTH
#             M_k3 = M + k3
            
#             if ((planet_object.planet_type == "LoFo14") or
#                (planet_object.planet_type == "ChRo16")):
#                 M_env_k3 = M_k3 - M_core
#                 f_env_k3 = (M_env_k3 / M_k3) * 100
#             elif planet_object.planet_type == "Ot20":
#                 R_k3 = plmoOt20.calculate_radius_planet_Ot20(M_k3)

#             if (i == 1) and (j == 1) and (M_k3 < M_core):
#                 # then I am still in the first RK iteration, and the initial
#                 # step size was likely too large -> set step size to minumum
#                 dt = min_step_size
#                 j += 1
#                 break
                
#             if ((planet_object.planet_type == "LoFo14") or
#                (planet_object.planet_type == "ChRo16")):
#                 Mdot4 = mass_loss_rate(t_=t + dt,
#                                        track_dict=track_dict,
#                                        planet_object=planet_object,
#                                        mass_loss_calc=mass_loss_calc,
#                                        epsilon=epsilon,
#                                        K_on=K_on, beta_on=beta_on,
#                                        f_env=f_env_k3,
#                                        beta_cutoff=beta_cutoff,
#                                        relation_EUV=relation_EUV)
#             elif planet_object.planet_type == "Ot20":
#                 Mdot4 = mass_loss_rate(t_=t + dt,
#                                        track_dict=track_dict,
#                                        planet_object=planet_object,
#                                        mass_loss_calc=mass_loss_calc,
#                                        epsilon=epsilon,
#                                        K_on=K_on, beta_on=beta_on,
#                                        radius_at_t_=R_k3,
#                                        beta_cutoff=beta_cutoff,
#                                        relation_EUV=relation_EUV)

#             k4 = (dt * Myr_to_sec * Mdot4) / M_EARTH
#             # total mass lost after time-step dt
#             M_lost = (k1 + 2*k2 + 2*k3 + k4) / 6. 

#             # update next value of the planet mass
#             M_new = M + M_lost
#             M_env_new = M_new - M_core
#             #print(dt, M_new, plmoOt20.calculate_radius_planet_Ot20(M_new))
            
#             # now it is time to check if atmosphere is gone or 
#             # if planet is close to complete atmosphere removal
#             # for Otegi planets, check if M_new < M_core
#             if ((np.isnan(M_lost) == True) \
#                     or (np.iscomplex(M_new) == True) or (M_new < M_core)) \
#                     and (dt == min_step_size):
#                 # if M_lost = nan (i.e. planet evaporates in one of the RK
#                 # steps) OR the four RK steps finish and the new planet mass is
#                 # smaller or equal to the core mass, then the planet counts as
#                 # evaporated!
#                 # if this condition is satisfied and the step size is already
#                 # at a minimum, then we assume the current RK iteration would
#                 # remove all atmosphere and only the rocky core is left at
#                 # t_i+dt; this terminates the code and returns the final planet
#                 # properties

#                 #print("Atmosphere has evaportated! Only bare rocky core left!"\
#                 #       + " STOP this madness!")

#                 # since the the stop criterium is reached, we assume at t_i+1
#                 # the planet only consists of the bare rocky core with the
#                 # planet mass equal to the core mass and the planet radius
#                 # equal to the core radius
#                 t_arr.append(t_arr[-1]+dt)
#                 M_arr.append(M_core)
#                 R_arr.append(R_core)
#                 Lx_arr.append(lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))
#                 envelope_left = False  # set flag for complete env. removal
#                 j += 1
#                 break
                
#             elif ((np.isnan(M_lost) == True) \
#                     or (np.iscomplex(M_new) == True) or (M_new < M_core)) \
#                     and (dt > min_step_size) \
#                     and (close_to_evaporation == False):
#                 #print("close to evaporation")
#                 # planet close to evaporation, but since the step size is not
#                 # minimum yet, we set it to its minimum value and run the RK
#                 # iteration again (until above stopping condition is fulfilled)
#                 dt = min_step_size
#                 close_to_evaporation = True
#                 # this variable is for making sure the code does not run into
#                 # an infinite loop when the planet is close to evaporation.
#                 # Once this condition is set to True, the code continues with
#                 # a fixed min. step size and is no longer allowed to adjust it.
#                 j += 1
#                 break
            
#             # this part is new compared to the one used in the PAPER (there we
#             # used a fixed step size!)
#             # if you're still in the while loop at this point, then calculate
#             # new radius and check how drastic the radius change would be;
#             # adjust the step size if too drastic or too little
#             if planet_object.planet_type == "LoFo14":
#                 f_env_new = (M_env_new/M_new)*100 # in %
#                 R_new = plmoLoFo14.calculate_planet_radius(
#                                                 M_core, f_env_new, t,
#                                                 flux_at_planet_earth(
#                                                     planet_object.Lbol,
#                                                     planet_object.distance),
#                                                 planet_object.metallicity
#                                                 )
#             elif planet_object.planet_type == "ChRo16":
#                 f_env_new = (M_env_new/M_new)*100 # in %
#                 R_new = plmoChRo16.calculate_planet_radius(
#                                                 M_core,
#                                                 f_env_new,
#                                                 flux_at_planet_earth(
#                                                     planet_object.Lbol,
#                                                     planet_object.distance),
#                                                 t,
#                                                 planet_object.core_comp
#                                                 )
#             elif planet_object.planet_type == "Ot20":
#                 R_new = plmoOt20.calculate_radius_planet_Ot20(M_new)
            
#             #print("R_new, f_new: ", R_new, f_env_new)
#             #print(abs((R-R_new)/R)*100)
            
#             # only adjust step size if planet is not close to complete evaporat.
#             if (close_to_evaporation == False):
#                 #print("radius check")
#                 # then do the check on how much the radius changes
#                 # R(t_i) compared to R(t_i+dt);
#                 # if radius change is larger than 0.5%, make step size smaller
#                 # by factor 10 OR if radius change is smaller than 0.02%, make
#                 # step size bigger by factor 10, but not bigger or smaller than
#                 # max. or min. step size!
#                 # if radius change too much/little, do not write anything to
#                 # file, instead do RK iteration again with new step size

#                 R_change = abs((R-R_new)/R)*100 # radius change compared to
#                                                 # previous radius - in percent
            
            
            
#             # only adjust step size if planet is not close to complete evaporat.
#             if (close_to_evaporation == False):
#                 #print("radius check")
#                 # then do the check on how much the radius changes
#                 # R(t_i) compared to R(t_i+dt);
#                 # if radius change is larger than 0.5%, make step size smaller
#                 # by factor 10 OR if radius change is smaller than 0.02%, make
#                 # step size bigger by factor 10, but not bigger or smaller than
#                 # max. or min. step size!
#                 # if radius change too much/little, do not write anything to
#                 # file, instead do RK iteration again with new step size

#                 R_change = abs((R-R_new)/R)*100 # radius change compared to
#                                                 # previous radius - in percent
#                 if (R_change > 0.5) \
#                         and (t < track_dict["t_curr"]) \
#                         and (dt > min_step_size):
#                     dt = dt / 10.
#                     j += 1
#                     break

#                 #elif ((R_change < 1.) \
#                 #			and (R_change >=0.1)) \
#                 #			and (t < track_dict["t_curr"]) \
#                 #			and (dt > min_step_size):
#                 #    dt = dt / 10.
#                 #    break

#                 elif (R_change < (0.02)) \
#                         and (t < track_dict["t_curr"]) \
#                         and (dt < max_step_size):
#                     dt = dt * 10.
#                     j += 1
#                     break    

#                 # NOTE: in principle I can adjust the code such that these 
#                 # hardcoded parameters are different for early planet evolution
#                 # where much more is happening typically, and late planet
#                 # evolution where almost no change is occurring anymore
#                 elif (R_change > 0.5) \
#                         and (t >= track_dict["t_curr"]) \
#                         and (dt > min_step_size):
#                     dt = dt / 10.
#                     j += 1
#                     break

#                 elif (R_change < (0.02)) \
#                         and (t >= track_dict["t_curr"]) \
#                         and (dt < max_step_size):
#                     dt = dt * 10
#                     j += 1
#                     break

#                 else: # if radius change is ok
#                     # do sanity check: is new planet mass is still greater than
#                     # the core mass? ->then there is still some atmosphere left
#                     # in this case update params and go into next RK iteration
#                     if ((M + M_lost) - M_core) > 0: 
#                         M = M + M_lost  # new planet mass (M_lost is negative)
#                         t = t_arr[-1] + dt  # updated time value t_i_plus_1
#                         M_arr.append(M)
#                         t_arr.append(t)
#                         Lx_arr.append(lx_evo(t=t, track_dict=track_dict))

#                         # calculate new radius with new planet mass/envelope
#                         # mass fraction & one time step later
#                         if planet_object.planet_type == "LoFo14":
#                             # calculate new envelope mass fraction:
#                             M_env = M - M_core
#                             f_env = (M_env/M)*100 # in %
#                             R = plmoLoFo14.calculate_planet_radius(
#                                                 M_core, f_env_new, t,
#                                                 flux_at_planet_earth(
#                                                     planet_object.Lbol,
#                                                     planet_object.distance),
#                                                 planet_object.metallicity
#                                                 )
#                         elif planet_object.planet_type == "ChRo16":
#                             # calculate new envelope mass fraction:
#                             M_env = M - M_core
#                             f_env = (M_env/M)*100 # in %
#                             R = plmoChRo16.calculate_planet_radius(
#                                                 M_core,
#                                                 f_env_new,
#                                                 flux_at_planet_earth(
#                                                     planet_object.Lbol,
#                                                     planet_object.distance),
#                                                 t,
#                                                 planet_object.core_comp
#                                                 )
#                         elif planet_object.planet_type == "Ot20":
#                             R = plmoOt20.calculate_radius_planet_Ot20(M)

#                         R_arr.append(R)
#                         i += 1 # update step to i+1
#                         j += 1

#                     else:
#                         # this should never happen
#                         sys.exit('sth went wrong of you see this!')
#                 break
            
#             elif (close_to_evaporation == True): 
#                 # if this condition is true, do not adjust step size
#                 # based on the radius change
#                 if ((M + M_lost) - M_core) > 0:
#                     M = M + M_lost # new planet mass (M_lost is negative)
#                     t = t_arr[-1] + dt #  updated time value t_i_plus_1
#                     M_arr.append(M)
#                     t_arr.append(t)
#                     Lx_arr.append(lx_evo(t=t, track_dict=track_dict))

#                     # calculate new radius with new planet mass/envelope
#                     # mass fraction & one time step later
#                     if planet_object.planet_type == "LoFo14":
#                         # calculate new envelope mass fraction:
#                         M_env = M - M_core
#                         f_env = (M_env/M)*100 # in %
#                         R = plmoLoFo14.calculate_planet_radius(
#                                             M_core, f_env_new, t,
#                                             flux_at_planet_earth(
#                                                 planet_object.Lbol,
#                                                 planet_object.distance),
#                                             planet_object.metallicity
#                                             )
#                     elif planet_object.planet_type == "ChRo16":
#                         # calculate new envelope mass fraction:
#                         M_env = M - M_core
#                         f_env = (M_env/M)*100 # in %
#                         R = plmoChRo16.calculate_planet_radius(
#                                             M_core,
#                                             f_env_new,
#                                             flux_at_planet_earth(
#                                                 planet_object.Lbol,
#                                                 planet_object.distance),
#                                             t,
#                                             planet_object.core_comp
#                                             )
#                     elif planet_object.planet_type == "Ot20":
#                         R = plmoOt20.calculate_radius_planet_Ot20(M)
                        
#                     R_arr.append(R)
#                     i += 1 # update step to i+1
#                     j += 1

#                 else:
#                     sys.exit('sth went wrong of you see this!')
#             break

#         if (envelope_left == False):
#             # planet has evaporated, so return last planet params for bare core
#             return np.array(t_arr), np.array(M_arr), \
#                    np.array(R_arr), np.array(Lx_arr)

#     #print("Done!")
#     return np.array(t_arr), np.array(M_arr), \
#            np.array(R_arr), np.array(Lx_arr)





















#####################################################################
# CURRENT IMPLEMENTATION
#####################################################################

def mass_planet_RK4_forward_LO14(epsilon, K_on, beta_on, planet_object,
                                 initial_step_size, t_final, track_dict,
                                 beta_cutoff=False, relation_EUV="Linsky"):
    """USED: 4th order Runge-Kutta as numerical integration  method 
    Integrate from the current time (t_start (where planet has R0 and M0) 
    into the future taking into account photoevaporative mass loss. 

    Parameters:
    -----------
    epsilon (float): evaporation efficiency
    K_on (str): set use of K parameter on or off ("on" or "off)
    beta_on (str): set use of beta parameter on or off ("on" or "off)
    planet_object: object of planet class which contains also stellar 
                   parameters and info about stellar evo track
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
    R_arr (array): radius array with radius evolution over time (from
                   thermal contraction and photoevaporative mass-loss)
    Lx_arr (array): array to trace the X-ray luminosity (mainly for
                    consistency checks)
    """ 
    # define some constants
    M_EARTH = const.M_earth.cgs.value
    Myr_to_sec = 1e6*365*86400
    
    # initialize the starting values for the X-ray luminosity at t_start,
    # the starting planet parameters, as well as beta & K (at t_start);
    # convert X-ray to XUV lum. and calculate planet's high energy 
    # incident flux
    Lx0 = lx_evo(t=track_dict["t_start"], track_dict=track_dict)
    Lxuv0 = l_xuv_all(Lx0, relation_EUV, planet_object.mass_star)
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
    # So when the planet mass gets smaller or equal to the core mass, we
    # assume only the bare rocky core is left.
        
    # specify beta and K
    if beta_on == "yes":
        # if XUV radius (as given by the Salz-approximation) is larger than
        # the planetary Roche lobe radius, we set beta such that R_XUV == R_RL
        R_RL = planet_object.distance * (const.au/const.R_earth) * (M0 / 3 * (M0 + \
               (planet_object.mass_star * (const.M_sun/const.M_earth))**(1/3)))
        beta = bk.beta_fct(M0, Fxuv0, R0, beta_cutoff)
        if (beta * R0) > R_RL:
            beta = R_RL / R0
    elif beta_on == "no":
        beta = 1.
    elif beta_on == "yes_old":
        # beta without the Roche-lobe cut-off
        beta = bk.beta_fct(M0, Fxuv0, R0, beta_cutoff)
        
    if K_on == "yes":
        K = K0 = bk.K_fct(planet_object.distance, M0,
                          planet_object.mass_star, R0)
    elif K_on == "no":
        K = K0 = 1.

    # since the step size is adaptive, I use lists to keep track of
    # the time, mass, radius and Lx evolution
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
    j = 1  # counter to track how many RK iterations have been attempted.
    envelope_left = True  # variable to flag a planet if envelope is gone
    close_to_evaporation = False  # variable to flag if planet is close to
                                  # complete atmospheric removal

    # make list with all step sizes, even those which resulted in too drastic 
    # radius changes -> then I can check if the code gets stuck in an infinite
    # loop between make_bigger, make_smaller, make_bigger, etc..
    step_size_list = []
    
    while t <= t_final:
        #print(i, j, " t= ", t, dt)

        # This step (Lx(t) calculation) is just for me to trace Lx and check 
        # if it is correct. It is NOT required since the Lx(t) calculation is 
        # embedded in the mass_loss_rate_fancy function)
        Lx_i = lx_evo(t=t, track_dict=track_dict)

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
        # 2) at the end of planet evolution when the planet mass gets very
        # close to the core mass, at some point the mass lost is larger than
        # the renmaining atmosphere mass (either at the end of a coplete RK
        # iteration, or this already happends in one of the RK substeps, in
        # which case the mass lost after the complete RK step evaluates to nan
        # and no new planet radius can be calculated). In both cases the planet
        # is assumed to be fully evaporated at t_i + dt.

        # add the current step size
        step_size_list.append(dt)
        # take the last 20 entries in step_size_array and check for 
        # constant back-and-forth between two step sizes
        # e.g. [0.1, 1.0, 0.1, 1.0, ...]
        # check if difference is the same, but values in array are not all 
        # the same; also need to check if every 2nd element is the same, 
        # same goes for 2(n+1)th
        step_size_array = np.array(step_size_list)
        step_size_difference = abs(np.diff(step_size_array[-20:]))
        if (len(step_size_array) >= 20): # check only after a 20 iterations
            if (np.all(step_size_difference == step_size_difference[0]) and 
                    ~np.all(step_size_array == step_size_array[0]) and 
                    np.all(step_size_array[::2] == step_size_array[0]) and 
                    np.all(step_size_array[1::2] == step_size_array[1])):
                print("no convergence, set min. step size.")
                # if no convergence, switch to minumum step size
                dt = min_step_size
        
        # else, all is good, continue with current step size
        while (envelope_left == True):
            # go through RK iterations as long as there is envelope left
            
            # apply Runge Kutta 4th order to find next value of M_dot
            # NOTE: the mass lost in one timestep is in Earth masses
            Mdot1 = mass_loss_rate_forward_LO14(t, epsilon, K_on,
                                                beta_on, planet_object,
                                                f_env, track_dict,
                                                beta_cutoff, relation_EUV)
            k1 = (dt * Myr_to_sec * Mdot1) / M_EARTH 
            M_05k1 = M + 0.5 * k1 # mass after 1st RK step
            M_env_05k1 = M_05k1 - M_core
            f_env_05k1 = (M_env_05k1 / M_05k1) * 100 # new envelope mass frac.
            if (i == 1) and (j == 1) and (M_05k1 < M_core):
                # then I am still in the first RK iteration, and the initial
                # step size was likely too large -> set step size to minumum
                dt = min_step_size
                j += 1
                break
            
            Mdot2 = mass_loss_rate_forward_LO14(t+0.5*dt, epsilon, K_on,
                                                beta_on, planet_object,
                                                f_env_05k1, track_dict,
                                                beta_cutoff, relation_EUV)
            k2 = (dt * Myr_to_sec * Mdot2) / M_EARTH
            M_05k2 = M + 0.5 * k2
            M_env_05k2 = M_05k2 - M_core
            f_env_05k2 = (M_env_05k2 / M_05k2) * 100
            if (i == 1) and (j == 1) and (M_05k2 < M_core):
                dt = min_step_size
                j += 1
                break

            Mdot3 = mass_loss_rate_forward_LO14(t + 0.5*dt, epsilon, K_on,
                                                beta_on, planet_object,
                                                f_env_05k2, track_dict,
                                                beta_cutoff, relation_EUV)
            k3 = (dt * Myr_to_sec * Mdot3) / M_EARTH
            M_k3 = M + k3
            M_env_k3 = M_k3 - M_core
            f_env_k3 = (M_env_k3 / M_k3) * 100
            if (i == 1) and (j == 1) and (M_k3 < M_core):
                dt = min_step_size
                j += 1
                break

            Mdot4 = mass_loss_rate_forward_LO14(t + dt, epsilon, K_on,
                                                beta_on, planet_object,
                                                f_env_k3, track_dict,
                                                beta_cutoff, relation_EUV)
            k4 = (dt * Myr_to_sec * Mdot4) / M_EARTH
            # total mass lost after time-step dt
            M_lost = (k1 + 2*k2 + 2*k3 + k4) / 6. 

            # update next value of the planet mass
            M_new = M + M_lost
            M_env_new = M_new - M_core

            # now it is time to check if atmosphere is gone or 
            # if planet is close to complete atmosphere removal
            if ((np.isnan(M_lost) == True) \
                    or (np.iscomplex(M_new) == True)) \
                    and (dt == min_step_size):
                # if M_lost = nan (i.e. planet evaporates in one of the RK
                # steps) OR the four RK steps finish and the new planet mass is
                # smaller or equal to the core mass, then the planet counts as
                # evaporated!
                # if this condition is satisfied and the step size is already
                # at a minimum, then we assume the current RK iteration would
                # remove all atmosphere and only the rocky core is left at
                # t_i+dt; this terminates the code and returns the final planet
                # properties

                #print("Atmosphere has evaportated! Only bare rocky core left!"\
                #       + " STOP this madness!")

                # since the the stop criterium is reached, we assume at t_i+1
                # the planet only consists of the bare rocky core with the
                # planet mass equal to the core mass and the planet radius
                # equal to the core radius
                t_arr.append(t_arr[-1]+dt)
                M_arr.append(M_core)
                R_arr.append(R_core)
                Lx_arr.append(lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))
                envelope_left = False  # set flag for complete env. removal
                j += 1
                break
                
            elif ((np.isnan(M_lost) == True) \
                    or (np.iscomplex(M_new) == True)) \
                    and (dt > min_step_size) \
                    and (close_to_evaporation == False):
                #print("close to evaporation")
                # planet close to evaporation, but since the step size is not
                # minimum yet, we set it to its minimum value and run the RK
                # iteration again (until above stopping condition is fulfilled)
                dt = min_step_size
                close_to_evaporation = True 
                # this variable is for making sure the code does not run into
                # an infinite loop when the planet is close to evaporation.
                # Once this condition is set to True, the code continues with
                # a fixed min. step size and is no longer allowed to adjust it.
                j += 1
                break
            
            # this part is new compared to the one used in the PAPER (there we
            # used a fixed step size!)
            # if you're still in the while loop at this point, then calculate
            # new radius and check how drastic the radius change would be;
            # adjust the step size if too drastic or too little
            f_env_new = (M_env_new/M_new)*100 # in %
            R_new = plmoLoFo14.calculate_planet_radius(
                                            M_core, f_env_new, t,
                                            planet_object.flux,
                                            planet_object.metallicity
                                            )
            #print("R_new, f_new: ", R_new, f_env_new)
            #print(abs((R-R_new)/R)*100)
            
            # only adjust step size if planet is not close to complete evaporat.
            if (close_to_evaporation == False):
                #print("radius check")
                # then do the check on how much the radius changes
                # R(t_i) compared to R(t_i+dt);
                # if radius change is larger than 0.5%, make step size smaller
                # by factor 10 OR if radius change is smaller than 0.02%, make
                # step size bigger by factor 10, but not bigger or smaller than
                # max. or min. step size!
                # if radius change too much/little, do not write anything to
                # file, instead do RK iteration again with new step size

                R_change = abs((R-R_new)/R)*100 # radius change compared to
                                                # previous radius - in percent
                if (R_change > 0.5) \
                        and (t < track_dict["t_curr"]) \
                        and (dt > min_step_size):
                    dt = dt / 10.
                    j += 1
                    break

                #elif ((R_change < 1.) \
                #			and (R_change >=0.1)) \
                #			and (t < track_dict["t_curr"]) \
                #			and (dt > min_step_size):
                #    dt = dt / 10.
                #    break

                elif (R_change < (0.02)) \
                        and (t < track_dict["t_curr"]) \
                        and (dt < max_step_size):
                    dt = dt * 10.
                    j += 1
                    break    

                # NOTE: in principle I can adjust the code such that these 
                # hardcoded parameters are different for early planet evolution
                # where much more is happening typically, and late planet
                # evolution where almost no change is occurring anymore
                elif (R_change > 0.5) \
                        and (t >= track_dict["t_curr"]) \
                        and (dt > min_step_size):
                    dt = dt / 10.
                    j += 1
                    break

                elif (R_change < (0.02)) \
                        and (t >= track_dict["t_curr"]) \
                        and (dt < max_step_size):
                    dt = dt * 10
                    j += 1
                    break

                else: # if radius change is ok
                    # do sanity check: is new planet mass is still greater than
                    # the core mass? ->then there is still some atmosphere left
                    # in this case update params and go into next RK iteration
                    if ((M + M_lost) - M_core) > 0: 
                        M = M + M_lost  # new planet mass (M_lost is negative)
                        t = t_arr[-1] + dt  # updated time value t_i_plus_1
                        M_arr.append(M)
                        t_arr.append(t)
                        Lx_arr.append(lx_evo(t=t, track_dict=track_dict))

                        # calculate new envelope mass fraction:
                        M_env = M - M_core
                        f_env = (M_env/M)*100 # in %

                        # calculate new radius with new planet mass/envelope
                        # mass fraction & one time step later
                        R = plmoLoFo14.calculate_planet_radius(
                                                M_core, f_env, t,
                                                planet_object.flux,
                                                planet_object.metallicity
                                                )
                        R_arr.append(R)
                        i += 1 # update step to i+1
                        j += 1

                    else:
                        # this should never happen
                        sys.exit('sth went wrong of you see this!')
                break
            
            elif (close_to_evaporation == True): 
                # if this condition is true, do not adjust step size
                # based on the radius change
                if ((M + M_lost) - M_core) > 0:
                    M = M + M_lost # new planet mass (M_lost is negative)
                    t = t_arr[-1] + dt #  updated time value t_i_plus_1
                    M_arr.append(M)
                    t_arr.append(t)
                    Lx_arr.append(lx_evo(t=t, track_dict=track_dict))

                    # calculate new envelope mass fraction:
                    M_env = M - M_core
                    f_env = (M_env/M)*100 # in %

                    # calculate new radius with new planet mass/envelope mass 
                    # fraction & one time step later
                    R = plmoLoFo14.calculate_planet_radius(
                                            M_core, f_env, t,
                                            planet_object.flux,
                                            planet_object.metallicity
                                            )
                    R_arr.append(R)
                    i += 1 # update step to i+1
                    j += 1

                else:
                    sys.exit('sth went wrong of you see this!')
            break

        if (envelope_left == False):
            # planet has evaporated, so return last planet params for bare core
            return np.array(t_arr), np.array(M_arr), \
                   np.array(R_arr), np.array(Lx_arr)

    #print("Done!")
    return np.array(t_arr), np.array(M_arr), \
           np.array(R_arr), np.array(Lx_arr)


def mass_planet_RK4_forward_LO14_HBA(planet_object,
                                     initial_step_size,
                                     t_final,
                                     track_dict,
                                     relation_EUV="Linsky"):
    """USED: 4th order Runge-Kutta as numerical integration  method 
    Integrate from the current time (t_start (where planet has R0 and M0) 
    into the future taking into account photoevaporative mass loss. 

    Parameters:
    -----------
    epsilon (float): evaporation efficiency
    planet_object: object of planet class which contains also stellar 
                   parameters and info about stellar evo track
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
    R_arr (array): radius array with radius evolution over time (from
                   thermal contraction and photoevaporative mass-loss)
    Lx_arr (array): array to trace the X-ray luminosity (mainly for
                    consistency checks)
    """ 
    # define some constants
    M_EARTH = const.M_earth.cgs.value
    Myr_to_sec = 1e6*365*86400
    
    # initialize the starting values for the X-ray luminosity at t_start,
    # the starting planet parameters, as well as beta & K (at t_start);
    # convert X-ray to XUV lum. and calculate planet's high energy 
    # incident flux
    Lx0 = lx_evo(t=track_dict["t_start"], track_dict=track_dict)
    Lxuv0 = l_xuv_all(Lx0, relation_EUV, planet_object.mass_star)
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
    # So when the planet mass gets smaller or equal to the core mass, we
    # assume only the bare rocky core is left.

    # since the step size is adaptive, I use lists to keep track of
    # the time, mass, radius and Lx evolution
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
    j = 1  # counter to track how many RK iterations have been attempted.
    envelope_left = True  # variable to flag a planet if envelope is gone
    close_to_evaporation = False  # variable to flag if planet is close to
                                  # complete atmospheric removal
    convergence_check = 0  # track how many times the convergence fails
    fix_step_size = False  # dummy variable to fix step size if 
                           # convergence fails too many times
    fix_step_size_N_times = False
    N_times = 0
    N = 50
        
    # make list with all step sizes, even those which resulted in too drastic 
    # radius changes -> then I can check if the code gets stuck in an infinite
    # loop between make_bigger, make_smaller, make_bigger, etc..
    step_size_list = []
    
    while t <= t_final:
        #print(i, j, " t= ", t, dt)

        # This step (Lx(t) calculation) is just for me to trace Lx and check 
        # if it is correct. It is NOT required since the Lx(t) calculation is 
        # embedded in the mass_loss_rate_fancy function)
        Lx_i = lx_evo(t=t, track_dict=track_dict)

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
        # 2) at the end of planet evolution when the planet mass gets very
        # close to the core mass, at some point the mass lost is larger than
        # the renmaining atmosphere mass (either at the end of a coplete RK
        # iteration, or this already happends in one of the RK substeps, in
        # which case the mass lost after the complete RK step evaluates to nan
        # and no new planet radius can be calculated). In both cases the planet
        # is assumed to be fully evaporated at t_i + dt.

        
        # add the current step size
        step_size_list.append(dt)
        # take the last 20 entries in step_size_array and check for 
        # constant back-and-forth between two step sizes
        # e.g. [0.1, 1.0, 0.1, 1.0, ...]
        # check if difference is the same, but values in array are not all 
        # the same; also need to check if every 2nd element is the same, 
        # same goes for 2(n+1)th
        step_size_array = np.array(step_size_list)
        x = 10 # take last x elements for comparison
        step_size_difference = abs(np.diff(step_size_array[-x:]))
        # check only after a 20 iterations
        if (len(step_size_array) >= 20) and (convergence_check <= 10) and (fix_step_size_N_times == False): 
            # check that all elements in step_size_difference are the same
            # check that not all elements in step_size_array the same
            # check that every other element is the same (even & odd)
            if (np.all(step_size_difference == step_size_difference[0]) 
                and ~np.all(step_size_array[-x:] == step_size_array[-x:][0])
                and np.all(step_size_array[-x:][::2] == step_size_array[-x:][0])
                and np.all(step_size_array[-x:][1::2] == step_size_array[-x:][1])):
                print("no convergence, set min. step size.")
                # if no convergence, switch to minumum step size
                # allow this only five times in one run!
                convergence_check += 1
                dt = min_step_size
                fix_step_size_N_times = True
                print("fix step size 50 times")
                
        elif (convergence_check >= 10):
            # only allow this check 10 times during one run
            dt = min_step_size
            print("reached")
            fix_step_size = True
        
        # fix step size only N times, then relax again to variable step size
        if (fix_step_size_N_times == True) and (N_times <= N):
            dt = min_step_size
            N_times += 1
            
        elif (N_times > N):
            print("reset N_times")
            #print(dt)
            fix_step_size_N_times = False
            N_times = 0
            
        #print(fix_step_size_N_times, N_times)
        
        # else, all is good, continue with current step size
        while (envelope_left == True):
            # go through RK iterations as long as there is envelope left
            
            # apply Runge Kutta 4th order to find next value of M_dot
            # NOTE: the mass lost in one timestep is in Earth masses
            Mdot1 = mass_loss_rate_forward_LO14_HBA(t, planet_object,
                                                    f_env, track_dict,
                                                    relation_EUV)
            k1 = (dt * Myr_to_sec * Mdot1) / M_EARTH 
            M_05k1 = M + 0.5 * k1 # mass after 1st RK step
            M_env_05k1 = M_05k1 - M_core
            f_env_05k1 = (M_env_05k1 / M_05k1) * 100 # new envelope mass frac.
            if (i == 1) and (j == 1) and (M_05k1 < M_core):
                # then I am still in the first RK iteration, and the initial
                # step size was likely too large -> set step size to minumum
                dt = min_step_size
                j += 1
                break
            
            Mdot2 = mass_loss_rate_forward_LO14_HBA(t+0.5*dt,planet_object,
                                                    f_env_05k1,track_dict,
                                                    relation_EUV)
            k2 = (dt * Myr_to_sec * Mdot2) / M_EARTH
            M_05k2 = M + 0.5 * k2
            M_env_05k2 = M_05k2 - M_core
            f_env_05k2 = (M_env_05k2 / M_05k2) * 100
            if (i == 1) and (j == 1) and (M_05k2 < M_core):
                dt = min_step_size
                j += 1
                break

            Mdot3 = mass_loss_rate_forward_LO14_HBA(t + 0.5*dt, planet_object,
                                                    f_env_05k2, track_dict,
                                                    relation_EUV)
            k3 = (dt * Myr_to_sec * Mdot3) / M_EARTH
            M_k3 = M + k3
            M_env_k3 = M_k3 - M_core
            f_env_k3 = (M_env_k3 / M_k3) * 100
            if (i == 1) and (j == 1) and (M_k3 < M_core):
                dt = min_step_size
                j += 1
                break

            Mdot4 = mass_loss_rate_forward_LO14_HBA(t + dt, planet_object,
                                                    f_env_k3, track_dict,
                                                    relation_EUV)
            k4 = (dt * Myr_to_sec * Mdot4) / M_EARTH
            # total mass lost after time-step dt
            M_lost = (k1 + 2*k2 + 2*k3 + k4) / 6. 

            # update next value of the planet mass
            M_new = M + M_lost
            M_env_new = M_new - M_core
            #print(dt, M_lost, M_new)

            # now it is time to check if atmosphere is gone or 
            # if planet is close to complete atmosphere removal
            if ((np.isnan(M_lost) == True) \
                    or (np.iscomplex(M_new) == True)) \
                    and (dt == min_step_size):
                # if M_lost = nan (i.e. planet evaporates in one of the RK
                # steps) OR the four RK steps finish and the new planet mass is
                # smaller or equal to the core mass, then the planet counts as
                # evaporated!
                # if this condition is satisfied and the step size is already
                # at a minimum, then we assume the current RK iteration would
                # remove all atmosphere and only the rocky core is left at
                # t_i+dt; this terminates the code and returns the final planet
                # properties

                #print("Atmosphere has evaportated! Only bare rocky core left!"\
                #       + " STOP this madness!")

                # since the the stop criterium is reached, we assume at t_i+1
                # the planet only consists of the bare rocky core with the
                # planet mass equal to the core mass and the planet radius
                # equal to the core radius
                t_arr.append(t_arr[-1]+dt)
                M_arr.append(M_core)
                R_arr.append(R_core)
                Lx_arr.append(lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))
                envelope_left = False  # set flag for complete env. removal
                j += 1
                break
                
            elif ((np.isnan(M_lost) == True) \
                    or (np.iscomplex(M_new) == True)) \
                    and (dt > min_step_size) \
                    and (close_to_evaporation == False):
                #print("close to evaporation")
                # planet close to evaporation, but since the step size is not
                # minimum yet, we set it to its minimum value and run the RK
                # iteration again (until above stopping condition is fulfilled)
                dt = min_step_size
                close_to_evaporation = True 
                # this variable is for making sure the code does not run into
                # an infinite loop when the planet is close to evaporation.
                # Once this condition is set to True, the code continues with
                # a fixed min. step size and is no longer allowed to adjust it.
                j += 1
                break
            
            # this part is new compared to the one used in the PAPER (there we
            # used a fixed step size!)
            # if you're still in the while loop at this point, then calculate
            # new radius and check how drastic the radius change would be;
            # adjust the step size if too drastic or too little
            f_env_new = (M_env_new/M_new)*100 # in %
            R_new = plmoLoFo14.calculate_planet_radius(
                                            M_core, f_env_new, t,
                                            planet_object.flux,
                                            planet_object.metallicity
                                            )
            #print(i, j, " t= ", t, dt)
            #print("R_new, f_new: ", R_new, f_env_new)
            #print(abs((R-R_new)/R)*100)
            
            # only adjust step size if planet is not close to complete
            # evaporation and no fixed step size is required
            if (close_to_evaporation == False) \
                and (fix_step_size == False) \
                and (fix_step_size_N_times == False):
                #print("radius check")
                # then do the check on how much the radius changes
                # R(t_i) compared to R(t_i+dt);
                # if radius change is larger than 0.5%, make step size smaller
                # by factor 10 OR if radius change is smaller than 0.02%, make
                # step size bigger by factor 10, but not bigger or smaller than
                # max. or min. step size!
                # if radius change too much/little, do not write anything to
                # file, instead do RK iteration again with new step size

                R_change = abs((R-R_new)/R)*100 # radius change compared to
                                                # previous radius - in percent
                if (R_change > 0.5) \
                        and (t < track_dict["t_curr"]) \
                        and (dt > min_step_size):
                    dt = dt / 10.
                    j += 1
                    break

                #elif ((R_change < 1.) \
                #			and (R_change >=0.1)) \
                #			and (t < track_dict["t_curr"]) \
                #			and (dt > min_step_size):
                #    dt = dt / 10.
                #    break

                elif (R_change < (0.02)) \
                        and (t < track_dict["t_curr"]) \
                        and (dt < max_step_size):
                    dt = dt * 10.
                    j += 1
                    break    

                # NOTE: in principle I can adjust the code such that these 
                # hardcoded parameters are different for early planet evolution
                # where much more is happening typically, and late planet
                # evolution where almost no change is occurring anymore
                elif (R_change > 0.5) \
                        and (t >= track_dict["t_curr"]) \
                        and (dt > min_step_size):
                    dt = dt / 10.
                    j += 1
                    break

                elif (R_change < (0.02)) \
                        and (t >= track_dict["t_curr"]) \
                        and (dt < max_step_size):
                    dt = dt * 10
                    j += 1
                    break

                else: # if radius change is ok
                    # do sanity check: is new planet mass is still greater than
                    # the core mass? ->then there is still some atmosphere left
                    # in this case update params and go into next RK iteration
                    if ((M + M_lost) - M_core) > 0: 
                        M = M + M_lost  # new planet mass (M_lost is negative)
                        t = t_arr[-1] + dt  # updated time value t_i_plus_1
                        M_arr.append(M)
                        t_arr.append(t)
                        Lx_arr.append(lx_evo(t=t, track_dict=track_dict))

                        # calculate new envelope mass fraction:
                        M_env = M - M_core
                        f_env = (M_env/M)*100 # in %

                        # calculate new radius with new planet mass/envelope
                        # mass fraction & one time step later
                        R = plmoLoFo14.calculate_planet_radius(
                                                M_core, f_env, t,
                                                planet_object.flux,
                                                planet_object.metallicity
                                                )
                        R_arr.append(R)
                        i += 1 # update step to i+1
                        j += 1

                    else:
                        # this should never happen
                        sys.exit('sth went wrong of you see this!')
                break
            
            elif (close_to_evaporation == True) \
                  or (fix_step_size == True) \
                  or (fix_step_size_N_times == True): 
                # if these conditions are true, do not adjust the
                # step size based on the radius change (keep fixed)
                if ((M + M_lost) - M_core) > 0:
                    M = M + M_lost # new planet mass (M_lost is negative)
                    t = t_arr[-1] + dt #  updated time value t_i_plus_1
                    M_arr.append(M)
                    t_arr.append(t)
                    Lx_arr.append(lx_evo(t=t, track_dict=track_dict))

                    # calculate new envelope mass fraction:
                    M_env = M - M_core
                    f_env = (M_env/M)*100 # in %

                    # calculate new radius with new planet mass/envelope mass 
                    # fraction & one time step later
                    R = plmoLoFo14.calculate_planet_radius(
                                            M_core, f_env, t,
                                            planet_object.flux,
                                            planet_object.metallicity
                                            )
                    R_arr.append(R)
                    i += 1 # update step to i+1
                    j += 1

                else:
                    sys.exit('sth went wrong of you see this!')
                
            break

        if (envelope_left == False):
            # planet has evaporated, so return last planet params for bare core
            return np.array(t_arr), np.array(M_arr), \
                   np.array(R_arr), np.array(Lx_arr)

    #print("Done!")
    return np.array(t_arr), np.array(M_arr), \
           np.array(R_arr), np.array(Lx_arr)



def mass_planet_RK4_forward_LO14_PAPER(epsilon, K_on, beta_on, 
                                       planet_object, initial_step_size,
                                       t_final, track_dict,
                                       relation_EUV="Linsky"):
    """USED: 4th order Runge-Kutta as numerical integration  method 
    Integrate from the current time (t_start (where planet has R0 and M0) 
    into the future taking into account photoevaporative mass loss. 
    Step size is fixed!

    Parameters:
    -----------
    epsilon (float): evaporation efficiency
    K_on (str): set use of K parameter on or off ("on" or "off)
    beta_on (str): set use of beta parameter on or off ("on" or "off)
    planet_object: object of planet class which contains also stellar 
                   parameters and info about stellar evo track
    step_size (float): fixed
    t_final (float): final time of simulation
    track_dict (dict): dictionary with Lx evolutionary track parameters

    Returns:
    --------
    t_arr (array): time array to trace mass and radius evolution
    M_arr (array): mass array with mass evolution over time (mass decrease)
    R_arr (array): radius array with radius evolution over time (from
                   thermal contraction and photoevaporative mass-loss)
    Lx_arr (array): array to trace the X-ray luminosity (mainly for
                    consistency checks)
    """ 
    
    M_EARTH = const.M_earth.cgs.value
    Myr_to_sec = 1e6*365*86400
    
    # initialize the starting values for Lxuv(t_start), mass, density, beta, K
    Lx0 = lx_evo(t=track_dict["t_start"], track_dict=track_dict)
    Lxuv0 = l_xuv_all(Lx0, relation_EUV, planet_object.mass_star)
    Fxuv0 = flux_at_planet(Lxuv0, planet_object.distance)
    
    # initial planet parameters at t_start
    f_env_0 = f_env = planet_object.fenv
    R0 = R = planet_object.radius
    M0 = M = planet_object.mass
    rho0 = rho = plmoLoFo14.density_planet(M0, R0)  # initial mean density
    M_env0 = M_env = M0 - planet_object.core_mass  # initial envelope mass
    M_core = planet_object.core_mass

    # specify beta and K
    if beta_on == "yes":
        # if XUV radius (as given by the Salz-approximation) is larger than
        # the planetary Roche lobe radius, we set beta such that R_XUV == R_RL
        R_RL = planet_object.distance * (const.au/const.R_earth) * (M0 / 3 * (M0 + \
               (planet_object.mass_star * (const.M_sun/const.M_earth))**(1/3)))
        beta = bk.beta_fct(M0, Fxuv0, R0, beta_cutoff)
        if (beta * R0) > R_RL:
            beta = R_RL / R0
    elif beta_on == "no":
        beta = 1.
    elif beta_on == "yes_old":
        # beta without the Roche-lobe cut-off
        beta = bk.beta_fct(M0, Fxuv0, R0, beta_cutoff)
        
    if K_on == "yes":
        K = K0 = bk.K_fct(planet_object.distance, M0,
        				  planet_object.mass_star, R0)
    elif K_on == "no":
        K = K0 = 1.
    
    t_start = track_dict["t_start"]
    t_max = t_final
    step_size = initial_step_size
    
    # create time array for integration (with user-specified step size)
    number = math.ceil((t_max-t_start)/step_size)
    times, step_size2 = np.linspace(t_start, t_max, number,
    								endpoint=True, retstep=True)
    dt = step_size2

    # here I make lists of all the values I would like to 
    # track & output in the end:
    M_arr = []
    M_arr.append(M0)
    R_arr = []
    R_arr.append(R0)
    t_arr = []
    t_arr.append(t_start)
    Lx_arr = []
    Lx_arr.append(Lx0)
    
    # CRITERION when to stop the mass-loss
    # the LofO14 planets have a FIXED core mass and thus core radius 
    # (bare rocky core)
    R_core = planet_object.core_radius # stop when this radius is reached!

    for i in range(0, len(times)-1):   
        #print(t_arr[i])

        # this is just for me to return the Lx(t) evolution to check if 
        # it is correct (not required since the Lx(t)
        # calculation is embedded in the mass_loss_rate_fancy function)
        Lx_i = lx_evo(t=t_arr[i], track_dict=track_dict)

        Mdot1 = mass_loss_rate_forward_LO14(times[i], epsilon,
        									K_on, beta_on, planet_object,
        									f_env, track_dict, beta_cutoff)
        k1 = (dt * Myr_to_sec * Mdot1) / M_EARTH # mass lost in one timestep 
        										 # in earth masses
        M_05k1 = M + 0.5 * k1
        M_env_05k1 = M_05k1 - M_core
        f_env_05k1 = (M_env_05k1 / M_05k1) * 100

        Mdot2 = mass_loss_rate_forward_LO14(times[i]+0.5*dt, epsilon,
        									K_on, beta_on, planet_object,
        									f_env_05k1, track_dict, beta_cutoff)
        k2 = (dt * Myr_to_sec * Mdot2) / M_EARTH
        M_05k2 = M + 0.5 * k2
        M_env_05k2 = M_05k2 - M_core
        f_env_05k2 = (M_env_05k2 / M_05k2) * 100

        Mdot3 = mass_loss_rate_forward_LO14(times[i]+0.5*dt, epsilon,
        									K_on, beta_on, planet_object,
        									f_env_05k2, track_dict, beta_cutoff)
        k3 = (dt * Myr_to_sec * Mdot3) / M_EARTH
        M_k3 = M + k3
        M_env_k3 = M_k3 - M_core
        f_env_k3 = (M_env_k3 / M_k3) * 100

        Mdot4 = mass_loss_rate_forward_LO14(times[i]+dt, epsilon,
        									K_on, beta_on, planet_object,
        									f_env_k3, track_dict, beta_cutoff)
        k4 = (dt * Myr_to_sec * Mdot4) / M_EARTH
        M_lost = (k1 + 2*k2 + 2*k3 + k4)/6. # after time-step dt

        # check if planet with new mass does still have some atmosphere
        if ((M + M_lost) - M_core) > 0:
            # then planet still has some atmosphere left -> continue
            M = M + M_lost # new planet mass
            M_env = M - M_core # new envelope mass
            t = t_arr[i] + dt  # t_i_plus_1 - update time value
            f_env = (M_env/M)*100 # in %
            # calculate new radius with new planet mass/envelope mass 
            # fraction & one time step later          
            R = plmoLoFo14.calculate_planet_radius(
            								M_core, f_env, t,
            								planet_object.flux,
            								planet_object.metallicity)
            t_arr.append(t)
            M_arr.append(M)
            R_arr.append(R)
            Lx_arr.append(lx_evo(t=t, track_dict=track_dict))

        else:
            # all atmosphere is gone -> terminate
            #print("Atmosphere has evaportated! Only bare rocky" \
            # 	  + "core left! STOP this madness!")

            # if the stop criterium is reached, I add the core mass 
            # and core radius to the array at t_i+1
            t_arr.append(t_arr[-1]+dt)
            M_arr.append(M_core)
            R_arr.append(R_core)
            Lx_arr.append(lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))
            return np.array(t_arr), np.array(M_arr), \
                   np.array(R_arr), np.array(Lx_arr)

    #print("Done!")
    return np.array(t_arr), np.array(M_arr), \
           np.array(R_arr), np.array(Lx_arr)


def mass_planet_RK4_forward_Ot20_HBA(planet_object,
                                     initial_step_size,
                                     t_final,
                                     track_dict,
                                     relation_EUV="Linsky"):
    """USED: 4th order Runge-Kutta as numerical integration  method 
    Integrate from the current time (t_start (where planet has R0 and M0)
    into the future taking into account photoevaporative mass loss. 
    
    Parameters:
    -----------
    epsilon (float): evaporation efficiency
    K_on (str): set use of K parameter on or off ("on" or "off)
    beta_on (str): set use of beta parameter on or off ("on" or "off)
    planet_object: object of planet class which contains also stellar 
                   parameters and info about stellar evo track
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
    R_arr (array): radius array with radius evolution over time (from
                   thermal contraction and photoevaporative mass-loss)
    Lx_arr (array): array to trace the X-ray luminosity (mainly for
                    consistency checks)
    """ 
    
    M_EARTH = const.M_earth.cgs.value
    Myr_to_sec = 1e6*365*86400
    
    # initialize the starting values for Lxuv(t_start), mass, density, beta, K
    Lx0 = lx_evo(t=track_dict["t_start"], track_dict=track_dict)
    Lxuv0 = l_xuv_all(Lx0, relation_EUV, planet_object.mass_star)
    Fxuv0 = flux_at_planet(Lxuv0, planet_object.distance)

    # "make" initial planet at t_start
    R0 = R = planet_object.radius
    M0 = M = planet_object.mass
    rho0 = rho = plmoOt20.density_planet(M0, R0)  # initial approx. density
    
    # here I make lists of all the values I would like to 
    # track & output in the end:
    M_arr = []
    M_arr.append(M0)
    R_arr = []
    R_arr.append(R0) 
    t_arr = []
    t0 = t = track_dict["t_start"]
    t_arr.append(t0) 
    Lx_arr = []
    Lx_arr.append(Lx0)
    
    # CRITERION when to stop the mass-loss
    # stop when this hardcoded radius is reached! 
    # (this is the minimum radius for which the volatile regime is valid)
    R_core = 2.15
    M_core = plmoOt20.calculate_mass_planet_Ot20(R_core)
    
    dt = initial_step_size
    # NOTE: minimum and maximum step size are HARDCODED for now (see further 
    # down in code for more details)
    min_step_size, max_step_size = 1e-2,  10.

    i = 1  # counter to track how many traced RK iterations have been performed
    j = 1  # counter to track how many RK iterations have been attempted.
    envelope_left = True  # variable to flag a planet if envelope is gone
    close_to_evaporation = False  # variable to flag if planet is close to
                                  # complete atmospheric removal
    convergence_check = 0  # track how many times the convergence fails
    fix_step_size = False  # dummy variable to fix step size if 
                           # convergence fails too many times
    fix_step_size_N_times = False
    N_times = 0
    N = 50

    # make list with all step sizes, even those which resulted in too drastic 
    # radius changes -> then I can check if the code gets stuck in an infinite
    # loop between make_bigger, make_smaller, make_bigger, etc..
    step_size_list = []
    
    while t <= t_final:
        #print(i, j, " t= ", t, dt)

        # This step (Lx(t) calculation) is just for me to trace Lx and check 
        # if it is correct. It is NOT required since the Lx(t) calculation is 
        # embedded in the mass_loss_rate_fancy function)
        Lx_i = lx_evo(t=t, track_dict=track_dict)

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
        # 2) at the end of planet evolution when the planet mass gets very
        # close to the core mass, at some point the mass lost is larger than
        # the renmaining atmosphere mass (either at the end of a coplete RK
        # iteration, or this already happends in one of the RK substeps, in
        # which case the mass lost after the complete RK step evaluates to nan
        # and no new planet radius can be calculated). In both cases the planet
        # is assumed to be fully evaporated at t_i + dt.

        # add the current step size
        step_size_list.append(dt)
        # take the last 20 entries in step_size_array and check for 
        # constant back-and-forth between two step sizes
        # e.g. [0.1, 1.0, 0.1, 1.0, ...]
        # check if difference is the same, but values in array are not all 
        # the same; also need to check if every 2nd element is the same, 
        # same goes for 2(n+1)th
        step_size_array = np.array(step_size_list)
        x = 10 # take last x elements for comparison
        step_size_difference = abs(np.diff(step_size_array[-x:]))
        # check only after a 20 iterations
        if (len(step_size_array) >= 20) and (convergence_check <= 10) and (fix_step_size_N_times == False): 
            # check that all elements in step_size_difference are the same
            # check that not all elements in step_size_array the same
            # check that every other element is the same (even & odd)
            if (np.all(step_size_difference == step_size_difference[0]) 
                and ~np.all(step_size_array[-x:] == step_size_array[-x:][0])
                and np.all(step_size_array[-x:][::2] == step_size_array[-x:][0])
                and np.all(step_size_array[-x:][1::2] == step_size_array[-x:][1])):
                print("no convergence, set min. step size.")
                # if no convergence, switch to minumum step size
                # allow this only five times in one run!
                convergence_check += 1
                dt = min_step_size
                fix_step_size_N_times = True
                print("fix step size 50 times")
                
        elif (convergence_check >= 10):
            # only allow this check 10 times during one run
            dt = min_step_size
            print("reached")
            fix_step_size = True
        
        # fix step size only N times, then relax again to variable step size
        if (fix_step_size_N_times == True) and (N_times <= N):
            dt = min_step_size
            N_times += 1
            
        elif (N_times > N):
            print("reset N_times")
            #print(dt)
            fix_step_size_N_times = False
            N_times = 0
        #print(fix_step_size_N_times, N_times)
        
        # else, all is good, continue with current step size
        while (envelope_left == True):
            # go through RK iterations as long as there is envelope left
            
            # apply Runge Kutta 4th order to find next value of M_dot
            # NOTE: the mass lost in one timestep is in Earth masses
            Mdot1 = mass_loss_rate_forward_Ot20_HBA(t, planet_object,
                                                    R, track_dict, beta_cutoff)
            k1 = (dt * Myr_to_sec * Mdot1) / M_EARTH 
            M_k1 = M + 0.5 * k1 # mass after 1st RK step
            R_k1 = plmoOt20.calculate_radius_planet_Ot20(M_k1)
            if (i == 1) and (j == 1) and (M_k1 < M_core):
                # then I am still in the first RK iteration, and the initial
                # step size was likely too large -> set step size to minumum
                dt = min_step_size
                j += 1
                break
            
            Mdot2 = mass_loss_rate_forward_Ot20_HBA(t+0.5*dt, planet_object,
                                                    R_k1, track_dict, beta_cutoff)
            k2 = (dt * Myr_to_sec * Mdot2) / M_EARTH
            M_05k2 = M + 0.5 * k2
            R_05k2 = plmoOt20.calculate_radius_planet_Ot20(M_05k2)
            if (i == 1) and (j == 1) and (M_05k2 < M_core):
                dt = min_step_size
                j += 1
                break

            Mdot3 = mass_loss_rate_forward_Ot20_HBA(t+0.5*dt, planet_object,
                                                    R_05k2, track_dict, beta_cutoff) 
            k3 = (dt * Myr_to_sec * Mdot3) / M_EARTH
            M_05k3 = M + k3
            R_05k3 = plmoOt20.calculate_radius_planet_Ot20(M_05k3)
            if (i == 1) and (j == 1) and (M_05k3 < M_core):
                dt = min_step_size
                j += 1
                break

            Mdot4 = mass_loss_rate_forward_Ot20_HBA(t+dt, planet_object,
                                                    R_05k3, track_dict, beta_cutoff) 
            k4 = (dt * Myr_to_sec * Mdot4) / M_EARTH
            # total mass lost after time-step dt
            M_lost = (k1 + 2*k2 + 2*k3 + k4) / 6. 

            # update next value of the planet mass
            M_new = M + M_lost

            # now it is time to check if atmosphere is gone or 
            # if planet is close to complete atmosphere removal
            if ((np.isnan(M_lost) == True) \
                    or (np.iscomplex(M_new) == True)) \
                    and (dt == min_step_size):
                # if M_lost = nan (i.e. planet evaporates in one of the RK
                # steps) OR the four RK steps finish and the new planet mass is
                # smaller or equal to the core mass, then the planet counts as
                # evaporated!
                # if this condition is satisfied and the step size is already
                # at a minimum, then we assume the current RK iteration would
                # remove all atmosphere and only the rocky core is left at
                # t_i+dt; this terminates the code and returns the final planet
                # properties

                #print("Atmosphere has evaportated! Only bare rocky core left!"\
                #       + " STOP this madness!")

                # since the the stop criterium is reached, we assume at t_i+1
                # the planet only consists of the bare rocky core with the
                # planet mass equal to the core mass and the planet radius
                # equal to the core radius
                t_arr.append(t_arr[-1]+dt)
                M_arr.append(M_core)
                R_arr.append(R_core)
                Lx_arr.append(lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))
                envelope_left = False  # set flag for complete env. removal
                j += 1
                break
                
            elif ((np.isnan(M_lost) == True) \
                    or (np.iscomplex(M_new) == True)) \
                    and (dt > min_step_size) \
                    and (close_to_evaporation == False):
                #print("close to evaporation")
                # planet close to evaporation, but since the step size is not
                # minimum yet, we set it to its minimum value and run the RK
                # iteration again (until above stopping condition is fulfilled)
                dt = min_step_size
                close_to_evaporation = True 
                # this variable is for making sure the code does not run into
                # an infinite loop when the planet is close to evaporation.
                # Once this condition is set to True, the code continues with
                # a fixed min. step size and is no longer allowed to adjust it.
                j += 1
                break
            
            # this part is new compared to the one used in the PAPER (there we
            # used a fixed step size!)
            # if you're still in the while loop at this point, then calculate
            # new radius and check how drastic the radius change would be;
            # adjust the step size if too drastic or too little
            R_new = plmoOt20.calculate_radius_planet_Ot20(M_new)
            #print("R_new, f_new: ", R_new, f_env_new)
            #print(abs((R-R_new)/R)*100)
            
            # only adjust step size if planet is not close to complete
            # evaporation and no fixed step size is required
            if (close_to_evaporation == False) \
                and (fix_step_size == False) \
                and (fix_step_size_N_times == False):
                #print("radius check")
                # then do the check on how much the radius changes
                # R(t_i) compared to R(t_i+dt);
                # if radius change is larger than 0.5%, make step size smaller
                # by factor 10 OR if radius change is smaller than 0.02%, make
                # step size bigger by factor 10, but not bigger or smaller than
                # max. or min. step size!
                # if radius change too much/little, do not write anything to
                # file, instead do RK iteration again with new step size

                R_change = abs((R-R_new)/R)*100 # radius change compared to
                                                # previous radius - in percent
                if (R_change > 1.) \
                        and (t < track_dict["t_curr"]) \
                        and (dt > min_step_size):
                    dt = dt / 10.
                    j += 1
                    break

                #elif ((R_change < 1.) \
                #           and (R_change >=0.1)) \
                #           and (t < track_dict["t_curr"]) \
                #           and (dt > min_step_size):
                #    dt = dt / 10.
                #    break

                elif (R_change < (0.01)) \
                        and (t < track_dict["t_curr"]) \
                        and (dt < max_step_size):
                    dt = dt * 10.
                    j += 1
                    break    

                # NOTE: in principle I can adjust the code such that these 
                # hardcoded parameters are different for early planet evolution
                # where much more is happening typically, and late planet
                # evolution where almost no change is occurring anymore
                elif (R_change > 1.) \
                        and (t >= track_dict["t_curr"]) \
                        and (dt > min_step_size):
                    dt = dt / 10.
                    j += 1
                    break

                elif (R_change < (0.01)) \
                        and (t >= track_dict["t_curr"]) \
                        and (dt < max_step_size):
                    dt = dt * 10
                    j += 1
                    break

                else: # if radius change is ok
                    # do sanity check: is new planet mass is still greater than
                    # the core mass? ->then there is still some atmosphere left
                    # in this case update params and go into next RK iteration
                    if ((M + M_lost) - M_core) > 0: 
                        M = M + M_lost  # new planet mass (M_lost is negative)
                        t = t_arr[-1] + dt  # updated time value t_i_plus_1
                        M_arr.append(M)
                        t_arr.append(t)
                        Lx_arr.append(lx_evo(t=t, track_dict=track_dict))

                        # calculate new envelope mass fraction:
                        M_env = M - M_core
                        f_env = (M_env/M)*100 # in %

                        # calculate new radius with new planet mass
                        # one time step later
                        R = plmoOt20.calculate_radius_planet_Ot20(M) 
                        R_arr.append(R)
                        i += 1 # update step to i+1
                        j += 1

                    else:
                        # this should never happen
                        sys.exit('sth went wrong of you see this!')
                break
            
            elif (close_to_evaporation == True) \
                  or (fix_step_size == True) \
                  or (fix_step_size_N_times == True): 
                # if these conditions are true, do not adjust the
                # step size based on the radius change (keep fixed)
                if ((M + M_lost) - M_core) > 0:
                    M = M + M_lost # new planet mass (M_lost is negative)
                    t = t_arr[-1] + dt #  updated time value t_i_plus_1
                    M_arr.append(M)
                    t_arr.append(t)
                    Lx_arr.append(lx_evo(t=t, track_dict=track_dict))

                    # calculate new envelope mass fraction:
                    M_env = M - M_core
                    f_env = (M_env/M)*100 # in %

                    # calculate new radius with new planet mass
                    # one time step later
                    R = plmoOt20.calculate_radius_planet_Ot20(M) 
                    R_arr.append(R)
                    i += 1 # update step to i+1
                    j += 1

                else:
                    sys.exit('sth went wrong of you see this!')
            break

        if (envelope_left == False):
            # planet has evaporated, so return last planet params for bare core
            return np.array(t_arr), np.array(M_arr), \
                   np.array(R_arr), np.array(Lx_arr)

    #print("Done!")
    return np.array(t_arr), np.array(M_arr), \
           np.array(R_arr), np.array(Lx_arr)


def mass_planet_RK4_forward_Ot20(epsilon, K_on, beta_on, planet_object,
								 initial_step_size, t_final, track_dict,
                                 relation_EUV="Linsky"):
    """USED: 4th order Runge-Kutta as numerical integration  method 
    Integrate from the current time (t_start (where planet has R0 and M0)
    into the future taking into account photoevaporative mass loss. 
    
    Parameters:
    -----------
    epsilon (float): evaporation efficiency
    K_on (str): set use of K parameter on or off ("on" or "off)
    beta_on (str): set use of beta parameter on or off ("on" or "off)
    planet_object: object of planet class which contains also stellar 
                   parameters and info about stellar evo track
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
    R_arr (array): radius array with radius evolution over time (from
                   thermal contraction and photoevaporative mass-loss)
    Lx_arr (array): array to trace the X-ray luminosity (mainly for
                    consistency checks)
    """ 
    
    M_EARTH = const.M_earth.cgs.value
    Myr_to_sec = 1e6*365*86400
    
    # initialize the starting values for Lxuv(t_start), mass, density, beta, K
    Lx0 = lx_evo(t=track_dict["t_start"], track_dict=track_dict)
    Lxuv0 = l_xuv_all(Lx0, relation_EUV, planet_object.mass_star)
    Fxuv0 = flux_at_planet(Lxuv0, planet_object.distance)

    # "make" initial planet at t_start
    R0 = R = planet_object.radius
    M0 = M = planet_object.mass
    rho0 = rho = plmoOt20.density_planet(M0, R0)  # initial approx. density
 
    # specify beta and K
    if beta_on == "yes":
        # if XUV radius (as given by the Salz-approximation) is larger than
        # the planetary Roche lobe radius, we set beta such that R_XUV == R_RL
        R_RL = planet_object.distance * (const.au/const.R_earth) * (M0 / 3 * (M0 + \
               (planet_object.mass_star * (const.M_sun/const.M_earth))**(1/3)))
        beta = bk.beta_fct(M0, Fxuv0, R0, beta_cutoff)
        if (beta * R0) > R_RL:
            beta = R_RL / R0
    elif beta_on == "no":
        beta = 1.
    elif beta_on == "yes_old":
        # beta without the Roche-lobe cut-off
        beta = bk.beta_fct(M0, Fxuv0, R0, beta_cutoff)
        
    if K_on == "yes":
        K = K0 = bk.K_fct(planet_object.distance, M0,
        				  planet_object.mass_star, R0)
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

    # CRITERION when to stop the mass-loss
    # stop when this hardcoded radius is reached! 
    # (this is the minimum radius for which the volatile regime is valid)
    R_core = 2.15
    M_core = plmoOt20.calculate_mass_planet_Ot20(R_core)

    dt = initial_step_size
    # NOTE: minimum and maximum step size are HARDCODED for now (see further 
    # down in code for more details)
    min_step_size, max_step_size = 1e-2,  10.

    i = 1  # counter to track how many traced RK iterations have been performed
    j = 1  # counter to track how many RK iterations have been attempted.
    envelope_left = True  # variable to flag a planet if envelope is gone
    close_to_evaporation = False  # variable to flag if planet is close to
                                  # complete atmospheric removal

    # make list with all step sizes, even those which resulted in too drastic 
    # radius changes -> then I can check if the code gets stuck in an infinite
    # loop between make_bigger, make_smaller, make_bigger, etc..
    step_size_list = []
    
    while t <= t_final:
        #print(i, j, " t= ", t, dt)

        # This step (Lx(t) calculation) is just for me to trace Lx and check 
        # if it is correct. It is NOT required since the Lx(t) calculation is 
        # embedded in the mass_loss_rate_fancy function)
        Lx_i = lx_evo(t=t, track_dict=track_dict)

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
        # 2) at the end of planet evolution when the planet mass gets very
        # close to the core mass, at some point the mass lost is larger than
        # the renmaining atmosphere mass (either at the end of a coplete RK
        # iteration, or this already happends in one of the RK substeps, in
        # which case the mass lost after the complete RK step evaluates to nan
        # and no new planet radius can be calculated). In both cases the planet
        # is assumed to be fully evaporated at t_i + dt.

        # add the current step size
        step_size_list.append(dt)
        # take the last 20 entries in step_size_array and check for 
        # constant back-and-forth between two step sizes
        # e.g. [0.1, 1.0, 0.1, 1.0, ...]
        # check if difference is the same, but values in array are not all 
        # the same; also need to check if every 2nd element is the same, 
        # same goes for 2(n+1)th
        step_size_array = np.array(step_size_list)
        step_size_difference = abs(np.diff(step_size_array[-20:]))
        if (len(step_size_array) >= 20): # check only after a 20 iterations
            if (np.all(step_size_difference == step_size_difference[0]) and 
                    ~np.all(step_size_array == step_size_array[0]) and 
                    np.all(step_size_array[::2] == step_size_array[0]) and 
                    np.all(step_size_array[1::2] == step_size_array[1])):
                print("no convergence, set min. step size.")
                # if no convergence, switch to minumum step size
                dt = min_step_size
        
        # else, all is good, continue with current step size
        while (envelope_left == True):
            # go through RK iterations as long as there is envelope left
            
            # apply Runge Kutta 4th order to find next value of M_dot
            # NOTE: the mass lost in one timestep is in Earth masses
            Mdot1 = mass_loss_rate_forward_Ot20(t, epsilon,
                                                K_on, beta_on,
                                                planet_object, R,
                                                track_dict, beta_cutoff)
            k1 = (dt * Myr_to_sec * Mdot1) / M_EARTH 
            M_k1 = M + 0.5 * k1 # mass after 1st RK step
            R_k1 = plmoOt20.calculate_radius_planet_Ot20(M_k1)
            if (i == 1) and (j == 1) and (M_k1 < M_core):
                # then I am still in the first RK iteration, and the initial
                # step size was likely too large -> set step size to minumum
                dt = min_step_size
                j += 1
                break
            
            Mdot2 = mass_loss_rate_forward_Ot20(t+0.5*dt, epsilon,
                                                K_on, beta_on,
                                                planet_object, R_k1,
                                                track_dict, beta_cutoff)
            k2 = (dt * Myr_to_sec * Mdot2) / M_EARTH
            M_05k2 = M + 0.5 * k2
            R_05k2 = plmoOt20.calculate_radius_planet_Ot20(M_05k2)
            if (i == 1) and (j == 1) and (M_05k2 < M_core):
                dt = min_step_size
                j += 1
                break

            Mdot3 = mass_loss_rate_forward_Ot20(t+0.5*dt, epsilon,
                                                K_on, beta_on,
                                                planet_object, R_05k2,
                                                track_dict, beta_cutoff) 
            k3 = (dt * Myr_to_sec * Mdot3) / M_EARTH
            M_05k3 = M + k3
            R_05k3 = plmoOt20.calculate_radius_planet_Ot20(M_05k3)
            if (i == 1) and (j == 1) and (M_05k3 < M_core):
                dt = min_step_size
                j += 1
                break

            Mdot4 = mass_loss_rate_forward_Ot20(t+dt, epsilon,
                                                K_on, beta_on,
                                                planet_object, R_05k3,
                                                track_dict, beta_cutoff) 
            k4 = (dt * Myr_to_sec * Mdot4) / M_EARTH
            # total mass lost after time-step dt
            M_lost = (k1 + 2*k2 + 2*k3 + k4) / 6. 

            # update next value of the planet mass
            M_new = M + M_lost

            # now it is time to check if atmosphere is gone or 
            # if planet is close to complete atmosphere removal
            if ((np.isnan(M_lost) == True) \
                    or (np.iscomplex(M_new) == True)) \
                    and (dt == min_step_size):
                # if M_lost = nan (i.e. planet evaporates in one of the RK
                # steps) OR the four RK steps finish and the new planet mass is
                # smaller or equal to the core mass, then the planet counts as
                # evaporated!
                # if this condition is satisfied and the step size is already
                # at a minimum, then we assume the current RK iteration would
                # remove all atmosphere and only the rocky core is left at
                # t_i+dt; this terminates the code and returns the final planet
                # properties

                #print("Atmosphere has evaportated! Only bare rocky core left!"\
                #       + " STOP this madness!")

                # since the the stop criterium is reached, we assume at t_i+1
                # the planet only consists of the bare rocky core with the
                # planet mass equal to the core mass and the planet radius
                # equal to the core radius
                t_arr.append(t_arr[-1]+dt)
                M_arr.append(M_core)
                R_arr.append(R_core)
                Lx_arr.append(lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))
                envelope_left = False  # set flag for complete env. removal
                j += 1
                break
                
            elif ((np.isnan(M_lost) == True) \
                    or (np.iscomplex(M_new) == True)) \
                    and (dt > min_step_size) \
                    and (close_to_evaporation == False):
                #print("close to evaporation")
                # planet close to evaporation, but since the step size is not
                # minimum yet, we set it to its minimum value and run the RK
                # iteration again (until above stopping condition is fulfilled)
                dt = min_step_size
                close_to_evaporation = True 
                # this variable is for making sure the code does not run into
                # an infinite loop when the planet is close to evaporation.
                # Once this condition is set to True, the code continues with
                # a fixed min. step size and is no longer allowed to adjust it.
                j += 1
                break
            
            # this part is new compared to the one used in the PAPER (there we
            # used a fixed step size!)
            # if you're still in the while loop at this point, then calculate
            # new radius and check how drastic the radius change would be;
            # adjust the step size if too drastic or too little
            R_new = plmoOt20.calculate_radius_planet_Ot20(M_new)
            #print("R_new, f_new: ", R_new, f_env_new)
            #print(abs((R-R_new)/R)*100)
            
            # only adjust step size if planet is not close to complete evaporat.
            if (close_to_evaporation == False):
                #print("radius check")
                # then do the check on how much the radius changes
                # R(t_i) compared to R(t_i+dt);
                # if radius change is larger than 0.5%, make step size smaller
                # by factor 10 OR if radius change is smaller than 0.02%, make
                # step size bigger by factor 10, but not bigger or smaller than
                # max. or min. step size!
                # if radius change too much/little, do not write anything to
                # file, instead do RK iteration again with new step size

                R_change = abs((R-R_new)/R)*100 # radius change compared to
                                                # previous radius - in percent
                if (R_change > 1.) \
                        and (t < track_dict["t_curr"]) \
                        and (dt > min_step_size):
                    dt = dt / 10.
                    j += 1
                    break

                #elif ((R_change < 1.) \
                #           and (R_change >=0.1)) \
                #           and (t < track_dict["t_curr"]) \
                #           and (dt > min_step_size):
                #    dt = dt / 10.
                #    break

                elif (R_change < (0.01)) \
                        and (t < track_dict["t_curr"]) \
                        and (dt < max_step_size):
                    dt = dt * 10.
                    j += 1
                    break    

                # NOTE: in principle I can adjust the code such that these 
                # hardcoded parameters are different for early planet evolution
                # where much more is happening typically, and late planet
                # evolution where almost no change is occurring anymore
                elif (R_change > 1.) \
                        and (t >= track_dict["t_curr"]) \
                        and (dt > min_step_size):
                    dt = dt / 10.
                    j += 1
                    break

                elif (R_change < (0.01)) \
                        and (t >= track_dict["t_curr"]) \
                        and (dt < max_step_size):
                    dt = dt * 10
                    j += 1
                    break

                else: # if radius change is ok
                    # do sanity check: is new planet mass is still greater than
                    # the core mass? ->then there is still some atmosphere left
                    # in this case update params and go into next RK iteration
                    if ((M + M_lost) - M_core) > 0: 
                        M = M + M_lost  # new planet mass (M_lost is negative)
                        t = t_arr[-1] + dt  # updated time value t_i_plus_1
                        M_arr.append(M)
                        t_arr.append(t)
                        Lx_arr.append(lx_evo(t=t, track_dict=track_dict))

                        # calculate new envelope mass fraction:
                        M_env = M - M_core
                        f_env = (M_env/M)*100 # in %

                        # calculate new radius with new planet mass
                        # one time step later
                        R = plmoOt20.calculate_radius_planet_Ot20(M) 
                        R_arr.append(R)
                        i += 1 # update step to i+1
                        j += 1

                    else:
                        # this should never happen
                        sys.exit('sth went wrong of you see this!')
                break
            
            elif (close_to_evaporation == True): 
                # if this condition is true, do not adjust step size
                # based on the radius change
                if ((M + M_lost) - M_core) > 0:
                    M = M + M_lost # new planet mass (M_lost is negative)
                    t = t_arr[-1] + dt #  updated time value t_i_plus_1
                    M_arr.append(M)
                    t_arr.append(t)
                    Lx_arr.append(lx_evo(t=t, track_dict=track_dict))

                    # calculate new envelope mass fraction:
                    M_env = M - M_core
                    f_env = (M_env/M)*100 # in %

                    # calculate new radius with new planet mass
                    # one time step later
                    R = plmoOt20.calculate_radius_planet_Ot20(M) 
                    R_arr.append(R)
                    i += 1 # update step to i+1
                    j += 1

                else:
                    sys.exit('sth went wrong of you see this!')
            break

        if (envelope_left == False):
            # planet has evaporated, so return last planet params for bare core
            return np.array(t_arr), np.array(M_arr), \
                   np.array(R_arr), np.array(Lx_arr)

    #print("Done!")
    return np.array(t_arr), np.array(M_arr), \
           np.array(R_arr), np.array(Lx_arr)


def mass_planet_RK4_forward_Ot20_PAPER(epsilon, K_on, beta_on, 
									   planet_object, initial_step_size, 
									   t_final, track_dict,
                                       relation_EUV="Linsky"):
    """USED: 4th order Runge-Kutta as numerical integration  method 
    Integrate from the current time (t_start (where planet has R0 and M0)
    into the future taking into account photoevaporative mass loss. 

    Input:
    ----------
    epsilon: evaporation efficiency
    K_on: set use of K parameter on or off 
    beta_on: set use of beta parameter on or off
    planet_object: object of planet class which contains also stellar
                   parameters and info about stellar evo track
    step_size: initial step_size, fixed
    t_final: final time of simulation
    track_dict: dictionary with Lx evolutionary track parameters
    
    Output:
    ----------
    t_arr, M_arr, R_arr, Lx_arr: array of time, mass, radius and 
    							 Lx values from t_start to t_final
    """ 
    
    M_EARTH = const.M_earth.cgs.value
    Myr_to_sec = 1e6*365*86400
    
    # initialize the starting values for Lxuv(t_start), mass, density, beta, K
    Lx0 = lx_evo(t=track_dict["t_start"], track_dict=track_dict)
    Lxuv0 = l_xuv_all(Lx0, relation_EUV, planet_object.mass_star)
    Fxuv0 = flux_at_planet(Lxuv0, planet_object.distance)
    
    # "make" initial planet at t_start
    R0 = R = planet_object.radius
    M0 = M = planet_object.mass
    rho0 = rho = plmoOt20.density_planet(M0, R0)  # initial approx. density
    
    # specify beta and K
    if beta_on == "yes":
        # if XUV radius (as given by the Salz-approximation) is larger than
        # the planetary Roche lobe radius, we set beta such that R_XUV == R_RL
        R_RL = planet_object.distance * (const.au/const.R_earth) * (M0 / 3 * (M0 + \
               (planet_object.mass_star * (const.M_sun/const.M_earth))**(1/3)))
        beta = bk.beta_fct(M0, Fxuv0, R0, beta_cutoff)
        if (beta * R0) > R_RL:
            beta = R_RL / R0
    elif beta_on == "no":
        beta = 1.
    elif beta_on == "yes_old":
        # beta without the Roche-lobe cut-off
        beta = bk.beta_fct(M0, Fxuv0, R0, beta_cutoff)
        
    if K_on == "yes":
        K = K0 = bk.K_fct(planet_object.distance, M0,
        				  planet_object.mass_star, R0)
    elif K_on == "no":
        K = K0 = 1.

    # create time array for integration (with user-specified step size)
    t_start, t_max = track_dict["t_start"], t_final
    step_size = initial_step_size
    number = math.ceil((t_max-t_start)/step_size)
    times, step_size2 = np.linspace(t_start, t_max, number,
    								endpoint=True, retstep=True)
    dt = step_size2

    # make lists of all the values wwe want to track & output in the end:
    M_arr = []
    M_arr.append(M0)
    R_arr = []
    R_arr.append(R0)
    t_arr = []
    t_arr.append(t_start)
    Lx_arr = []
    Lx_arr.append(Lx0)
    
    # CRITERION when to stop the mass-loss
    R_core = 2.15 # stop when this radius is reached! (this is the minimum
                  # radius for which the volatile regime is valid)
    M_core = plmoOt20.calculate_mass_planet_Ot20(R_core)

    for i in range(0, len(times)-1):   
        # this is just for me to return the Lx(t) evolution to check if it is
        # correct (not required since the Lx(t) calculation is embedded in 
        # the mass_loss_rate_fancy function)
        Lx_i = lx_evo(t=t_arr[i], track_dict=track_dict)

        Mdot1 = mass_loss_rate_forward_Ot20(times[i], epsilon, 
        									K_on, beta_on, 
        									planet_object, R, 
        									track_dict, beta_cutoff)
        k1 = (dt * Myr_to_sec * Mdot1) / M_EARTH # in earth masses
        M_05k1 = M + 0.5 * k1     
        R_05k1 = plmoOt20.calculate_radius_planet_Ot20(M_05k1)
        
        Mdot2 = mass_loss_rate_forward_Ot20(times[i]+0.5*dt, epsilon,
        									K_on, beta_on,
        									planet_object, R_05k1,
        									track_dict, beta_cutoff)
        k2 = (dt * Myr_to_sec * Mdot2) / M_EARTH
        M_05k2 = M + 0.5 * k2
        R_05k2 = plmoOt20.calculate_radius_planet_Ot20(M_05k2)
        
        Mdot3 = mass_loss_rate_forward_Ot20(times[i]+0.5*dt, epsilon,
        									K_on, beta_on,
        									planet_object, R_05k2,
        									track_dict, beta_cutoff) 
        k3 = (dt * Myr_to_sec * Mdot3) / M_EARTH
        M_05k3 = M + 0.5 * k3
        R_05k3 = plmoOt20.calculate_radius_planet_Ot20(M_05k3)
        
        Mdot4 = mass_loss_rate_forward_Ot20(times[i]+dt, epsilon,
        									K_on, beta_on,
        									planet_object, R_05k3,
        									track_dict, beta_cutoff) 
        k4 = (dt * Myr_to_sec * Mdot4) / M_EARTH
        M_lost = (k1 + 2*k2 + 2*k3 + k4)/6. # mass lost after time-step dt

        # check if planet with new mass does still have some atmosphere
        if ((M + M_lost) - M_core) > 0:
            # then planet still has some atmosphere left -> continue
            M = M + M_lost # new planet mass (M_lost is negative)
            # t_i_plus_1 - update time value
            t = t_arr[-1] + dt
            # calculate new radius with new planet mass
            R = plmoOt20.calculate_radius_planet_Ot20(M)
            M_arr.append(M)
            R_arr.append(R)
            t_arr.append(t)
            Lx_arr.append(lx_evo(t=t, track_dict=track_dict))
            
        else:
            # all atmosphere is gone (based on criterion set at the top)
            #print("Atmosphere has evaportated! Only bare rocky core" \
            #	 + "left! STOP this madness!")

            # if the stop criterium is reached, I add the core mass 
            # and core radius to the array at t_i+1
            
            t_arr.append(t_arr[-1]+dt)
            M_arr.append(M_core)
            R_arr.append(R_core)
            Lx_arr.append(lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))
            return np.array(t_arr), np.array(M_arr), \
                   np.array(R_arr), np.array(Lx_arr)
    
    # if planet survives, output the final arrays
    #Lx_arr[i+1] = lx_evo(t=t_arr[-1]+dt, track_dict=track_dict)
    #Lx_arr.append(lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))
    #print("Done!")
    return np.array(t_arr), np.array(M_arr), \
           np.array(R_arr), np.array(Lx_arr)
