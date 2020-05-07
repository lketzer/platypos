# highest level function

from Lx_evo_and_flux import Lx_evo, flux_at_planet_earth, L_xuv_all, flux_at_planet
import Planet_models_LoFo14 as plmo14
import Planet_model_Ot20 as plmoOt20
import Beta_K_functions as bk
import astropy.units as u
from astropy import constants as const
from Mass_loss_rate_function import mass_loss_rate_forward_LO14, mass_loss_rate_forward_Ot20
import numpy as np
import math


def mass_planet_RK4_forward_LO14(epsilon, K_on, beta_on, planet_object, initial_step_size, t_final, track_dict):
    """USED: Runge-Kutta method to numerically integrate an ordinary differential equation by using a trial step at the 
    midpoint of an interval to cancel out lower-order error terms.
    -> Integrate from the current time (t_start (where planet has R0 and M0) into the future. 
    *********I have sth like: Mdot = f(M_pl, Fxuv), and I want M_pl(t)*********
    Note: M_star should be defined outside this function.
    
    planet_object contains all the initial parameters of the system, including M_star 
    (but in solar units and this function needs it in cgs)
    
    Parameters:
    ----------
    Lx_evo:
    mass_loss_rate_fancy_LO14:
    calculate_planet_radius:
    epsilon:
    K_on:
    beta_on, 
    planet_object: planet object which contains also stellar parameters & info about stellar evo track
    f_env_0: initial envelope mass fraction)
    step_size: initial step_size (can be adjusted during calculation)
    t_final: final time of simulation
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
    rho0 = rho = plmo14.density_planet(M0, R0)  # initial mean density
    M_env0 = M_env = M0 - planet_object.core_mass # initial envelope mass
    M_core = planet_object.core_mass
    
#     print("Menv (in earth masses) = ", M_env/MEcgs)
#     print("M_0 = ", np.round(M0/MEcgs, 3))
#     print("R_0 = ", np.round(R0/REcgs, 3))
#     print("fenv (%) = ", np.round(f_env, 3))
    
    # specify beta0 and K0
    if beta_on == "yes": # use the beta estimate from Salz et al. 2016
        beta = beta0 = bk.beta_fct_LO14(M0, Fxuv0, R0)
    elif beta_on == "no":
        beta = beta0 = 1.
    if K_on == "yes": # use K-correction estimation from Erkaev et al. 2007
        K = K0 = bk.K_fct_LO14(planet_object.distance, M0, planet_object.mass_star, R0)
    elif K_on == "no":
        K = K0 = 1.
#     print("beta0 = ", beta, " - K0 = ", K)
    
    M_arr = []
    M_arr.append(M0)
    R_arr = []
    R_arr.append(R0) 
    t_arr = []
    t0 = t = track_dict["t_start"]
    t_arr.append(t0) 
    Lx_arr = []
    Lx_arr.append(Lx0)

    ################
    # CRITERION when to stop the mass-loss
    ################
    # for the Lopez planets I have a specified core mass and thus a fixed core radius (bare rocky core)
    R_core = planet_object.core_radius # stop when this radius is reached!
    
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
        Mdot1 = mass_loss_rate_forward_LO14(t, epsilon, K_on, beta_on, planet_object, f_env, track_dict)
        k1 = (dt*Myr_to_sec * Mdot1)/M_earth # mass lost in one timestep in earth masses
        # check if this new planet would still be larger than core radius!
        ########################################################
        # I think I should solve this differently!! TALK TO JORI
        ########################################################
        if np.iscomplex(k1):
            print("Atmosphere has evaportated! Only bare rocky core left! STOP this madness!")
            # if the stop criterium is reached, I add the core mass & core radius to the array at t_i+1
            print("t = ", t_arr[-1]+dt)
            t_arr.append(t_arr[-1]+dt)
            M_arr.append(M_core)
            print("M_core = ", M_core)
            R_arr.append(R_core)
            print("R_core = ", R_core)
            Lx_arr.append(Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))            
            return np.array(t_arr), np.array(M_arr), np.array(R_arr), np.array(Lx_arr)
        ###############################################################################################################
        M_05k1 = M + 0.5*k1     
        M_env_05k1 = M_05k1 - M_core
        f_env_05k1 = (M_env_05k1/M_05k1)*100 # new mass fraction  
        
        Mdot2 = mass_loss_rate_forward_LO14(t+0.5*dt, epsilon, K_on, beta_on, planet_object, f_env_05k1, track_dict)
        #R_p=radius_planet(M + 0.5*k1))
        k2 = (dt*Myr_to_sec * Mdot2)/M_earth
        if np.iscomplex(k2):
            print("Atmosphere has evaportated! Only bare rocky core left! STOP this madness!")
            # if the stop criterium is reached, I add the core mass & core radius to the array at t_i+1
            print("t = ", t_arr[-1]+dt)
            t_arr.append(t_arr[-1]+dt)
            M_arr.append(M_core)
            print("M_core = ", M_core)
            R_arr.append(R_core)
            print("R_core = ", R_core)
            Lx_arr.append(Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))            
            return np.array(t_arr), np.array(M_arr), np.array(R_arr), np.array(Lx_arr)
        ###############################################################################################################
        M_05k2 = M + 0.5*k2
        M_env_05k2 = M_05k2 - M_core
        f_env_05k2 = (M_env_05k2/M_05k2)*100 # new mass fraction
        
        Mdot3 = mass_loss_rate_forward_LO14(t+0.5*dt, epsilon, K_on, beta_on, planet_object, f_env_05k2, track_dict) 
        # R_p=radius_planet(M + 0.5*k2))
        k3 = (dt*Myr_to_sec * Mdot3)/M_earth
        if np.iscomplex(k3):
            print("Atmosphere has evaportated! Only bare rocky core left! STOP this madness!")
            # if the stop criterium is reached, I add the core mass & core radius to the array at t_i+1
            print("t = ", t_arr[-1]+dt)
            t_arr.append(t_arr[-1]+dt)
            M_arr.append(M_core)
            print("M_core = ", M_core)
            R_arr.append(R_core)
            print("R_core = ", R_core)
            Lx_arr.append(Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))            
            return np.array(t_arr), np.array(M_arr), np.array(R_arr), np.array(Lx_arr)
        ###############################################################################################################
        M_k3 = M + k3
        M_env_k3 = M_k3 - M_core
        f_env_k3 = (M_env_k3/M_k3)*100 # new mass fraction
        
        Mdot4 = mass_loss_rate_forward_LO14(t+dt, epsilon, K_on, beta_on, planet_object, f_env_k3, track_dict) 
        # R_p=radius_planet(M + k3))
        k4 = (dt*Myr_to_sec * Mdot4)/M_earth
        if np.iscomplex(k4):
            print("Atmosphere has evaportated! Only bare rocky core left! STOP this madness!")
            # if the stop criterium is reached, I add the core mass & core radius to the array at t_i+1
            print("t = ", t_arr[-1]+dt)
            t_arr.append(t_arr[-1]+dt)
            M_arr.append(M_core)
            print("M_core = ", M_core)
            R_arr.append(R_core)
            print("R_core = ", R_core)
            Lx_arr.append(Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))            
            return np.array(t_arr), np.array(M_arr), np.array(R_arr), np.array(Lx_arr)
        ###############################################################################################################

        M_lost = (k1 + 2*k2 + 2*k3 + k4)/6. # after time-step dt
        ###########################################################################################
        # this part is new compared to the one used in the PAPER (there we used a fixed step size!)
        ###########################################################################################
        # if radius change is too drastic, decrease step size
        M_new = M + M_lost
        M_env_new = M_new - M_core
        f_env_new = (M_env_new/M_new)*100 # in %
        R_new = plmo14.calculate_planet_radius(M_core, f_env_new, t, flux_at_planet_earth(planet_object.Lbol,
                                                                                      planet_object.distance), 
                                               planet_object.metallicity)
        
        # adjust step size, if radius change is too drastic or too little
        if (abs((R-R_new)/R)*100 > 1) & (t < track_dict["t_curr"]) & (dt > 1e-2):#10**(-1)): # smaller than 1%
            dt = dt/10.
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
        
        else: # if radius change is ok, check if there is still envelope left
            if ((M + M_lost) - M_core) >= 0:
                # then planet still has some atmosphere left -> continue
                M = M + M_lost # new planet mass (M_lost is negative)
                M_arr.append(M)
                M_env = M - M_core # new envelope mass
                t = t_arr[-1] + dt #t_start + i*dt # t_i_plus_1 - update time value
                t_arr.append(t) # new time t
                Lx_arr.append(Lx_evo(t=t, track_dict=track_dict)) # Lx at new time t

                # new envelope mass fraction:
                f_env = (M_env/M)*100 # in %
                #print("f_env = ", f_env)
                # calculate new radius with new planet mass/envelope mass fraction & one time step later
                R = plmo14.calculate_planet_radius(M_core, f_env, t, flux_at_planet_earth(planet_object.Lbol,
                                                                                          planet_object.distance), 
                                                   planet_object.metallicity)
                R_arr.append(R)
                i = i+1 # update step to i+1
                
            else:
                # all atmosphere is gone -> terminate
                print("Atmosphere has evaportated! Only bare rocky core left! STOP this madness!")

                # if the stop criterium is reached, I add the core mass & core radius to the array at t_i+1
                print("t = ", t_arr[-1]+dt)
                t_arr.append(t_arr[-1]+dt)
                M_arr.append(M_core)
                print("M_core = ", M_core)
                R_arr.append(R_core)
                print("R_core = ", R_core)
                Lx_arr.append(Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))            
                return np.array(t_arr), np.array(M_arr), np.array(R_arr), np.array(Lx_arr)
                              
    print("Done!")
    return np.array(t_arr), np.array(M_arr), np.array(R_arr), np.array(Lx_arr)


##################################################################################################################################
##################################################################################################################################
##################################################################################################################################


def mass_planet_RK4_forward_LO14_PAPER(epsilon, K_on, beta_on, planet_object, initial_step_size, t_final, track_dict):
    """USED: Runge-Kutta method to numerically integrate an ordinary differential equation by using a trial step at the 
    midpoint of an interval to cancel out lower-order error terms.
    -> Integrate from the current time (t_start (where planet has R0 and M0) into the future. 
    *********I have sth like: Mdot = f(M_pl, Fxuv), and I want M_pl(t)*********
    Note: M_star should be defined outside this function.""" 
    
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
    rho0 = rho = plmo14.density_planet(M0, R0)  # initial mean density
    M_env0 = M_env = M0 - planet_object.core_mass # initial envelope mass
    M_core = planet_object.core_mass
    
    # specify beta0 and K0
    if beta_on == "yes": # use the beta estimate from Salz et al. 2016
        beta = beta0 = bk.beta_fct_LO14(M0, Fxuv0, R0)
    elif beta_on == "no":
        beta = beta0 = 1.
    if K_on == "yes": # use K-correction estimation from Erkaev et al. 2007
        K = K0 = bk.K_fct_LO14(planet_object.distance, M0, planet_object.mass_star, R0)
    elif K_on == "no":
        K = K0 = 1.
    
    t_start = track_dict["t_start"]
    t_max = t_final
    step_size = initial_step_size
    
    # create time array for integration (with user-specified step size)
    number = math.ceil((t_max-t_start)/step_size)
    times, step_size2 = np.linspace(t_start, t_max, number, endpoint=True, retstep=True)
    print('stepsize=', step_size2)
    
    # here I make lists of all the values I would like to track & output in the end:
    M_arr = [0]*len(times)
    M_arr[0] = M = M0 # inital value for Mdot at t_curr
    R_arr = [0]*len(times)
    R_arr[0] = R = R0
    t_arr = [0]*len(times)
    t_arr[0] = t_start  # inital value for t
    Lx_arr = [0]*len(times)
    #Lx_arr[0] = Lx0
    
    ################
    # CRITERIUM when to stop the mass-loss
    ################
    # STOP the mass loss when Mercury Radius is reached! (bare rocky core, nothing more to evaporate)
    #R_mercury_ = 0.3829*const.R_earth.cgs # true Mercury radius
    #M_mercury = 0.055*const.M_earth.cgs
    ################
    # I want to kill plants (which is approx. when they reach 2 R_earth, or as far as Otegi M-R relation for volatile regime is valid)
    ################
    #R_mercury_ = 2*const.R_earth.cgs
    # I use the M-R relation to get the radius that a planet with the mass of Mercury would have
    # R_mercury_ = radius_planet(M_mercury)
    # and when R_p < R_mercury_, Mdot = 0 [for now I stop when this condition is met.]
    ################
    
    ################
    # CRITERION when to stop the mass-loss
    ################
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
        
        ################
        ###############################################################################################################
        Mdot1 = mass_loss_rate_forward_LO14(times[i], epsilon, K_on, beta_on, planet_object, f_env, track_dict)
        k1 = (dt*Myr_to_sec * Mdot1)/M_earth # mass lost in one timestep in earth masses
        ###############################################################################################################
        M_05k1 = M + 0.5*k1     
        M_env_05k1 = M_05k1 - M_core
        f_env_05k1 = (M_env_05k1/M_05k1)*100 # new mass fraction  
        
        Mdot2 = mass_loss_rate_forward_LO14(times[i]+0.5*dt, epsilon, K_on, beta_on, planet_object, f_env_05k1, track_dict)
        #R_p=radius_planet(M + 0.5*k1))
        k2 = (dt*Myr_to_sec * Mdot2)/M_earth
        ###############################################################################################################
        M_05k2 = M + 0.5*k2
        M_env_05k2 = M_05k2 - M_core
        f_env_05k2 = (M_env_05k2/M_05k2)*100 # new mass fraction
        
        Mdot3 = mass_loss_rate_forward_LO14(times[i]+0.5*dt, epsilon, K_on, beta_on, planet_object, f_env_05k2, track_dict) 
        # R_p=radius_planet(M + 0.5*k2))
        k3 = (dt*Myr_to_sec * Mdot3)/M_earth
        ###############################################################################################################
        M_k3 = M + k3
        M_env_k3 = M_k3 - M_core
        f_env_k3 = (M_env_k3/M_k3)*100 # new mass fraction
        
        Mdot4 = mass_loss_rate_forward_LO14(times[i]+dt, epsilon, K_on, beta_on, planet_object, f_env_k3, track_dict) 
        # R_p=radius_planet(M + k3))
        k4 = (dt*Myr_to_sec * Mdot4)/M_earth
        ###############################################################################################################
        
        M_lost = (k1 + 2*k2 + 2*k3 + k4)/6. # after time-step dt ###############################################################################################################
        
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
            R_arr[i+1] = R = plmo14.calculate_planet_radius(M_core, f_env, t, flux_at_planet_earth(planet_object.Lbol,
                                                                                          planet_object.distance), 
                                                   planet_object.metallicity)
            
        else:
            # all atmosphere is gone -> terminate
            print("Atmosphere has evaportated! Only bare rocky core left! STOP this madness!")

            # if the stop criterium is reached, I add the core mass & core radius to the array at t_i+1
            t_arr = np.trim_zeros(t_arr)
            print(t_arr[-1]+dt)
            print(t_arr[i]+dt)
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


##################################################################################################################################
##################################################################################################################################
##################################################################################################################################


def mass_planet_RK4_forward_Ot14(epsilon, K_on, beta_on, planet_object, initial_step_size, t_final, track_dict):
    """USED: Runge-Kutta method to numerically integrate an ordinary differential equation by using a trial step at the 
    midpoint of an interval to cancel out lower-order error terms.
    -> Integrate from the current time (t_start (where planet has R0 and M0) into the future. 
    *********I have sth like: Mdot = f(M_pl, Fxuv), and I want M_pl(t)*********
    Note: M_star should be defined outside this function.
    
    planet_object contains all the initial parameters of the system, including M_star 
    (but in solar units and this function needs it in cgs)
    
    Parameters:
    ----------
    Lx_evo:
    mass_loss_rate_fancy_LO14:
    calculate_planet_radius:
    epsilon:
    K_on:
    beta_on, 
    planet_object: planet object which contains also stellar parameters & info about stellar evo track
    f_env_0: initial envelope mass fraction)
    step_size: initial step_size (can be adjusted during calculation)
    t_final: final time of simulation
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
        beta = beta0 = bk.beta_fct_LO14(M0, Fxuv0, R0)
    elif beta_on == "no":
        beta = beta0 = 1.
    if K_on == "yes": # use K-correction estimation from Erkaev et al. 2007
        K = K0 = bk.K_fct_LO14(planet_object.distance, M0, planet_object.mass_star, R0)
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
    
    ################
    # CRITERION when to stop the mass-loss
    ################
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
        # check if this new planet would still be larger than core radius!
        if np.iscomplex(k1):
            print("STOP this madness! Leaving volatile regime!")
            # if the stop criterium is reached, I add the core mass & core radius to the array at t_i+1
            print("t = ", t_arr[-1]+dt)
            t_arr.append(t_arr[-1]+dt)
            M_arr.append(plmoOt20.calculate_planet_mass_Ot20(R_core))
            R_arr.append(R_core)
            Lx_arr.append(Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))            
            return np.array(t_arr), np.array(M_arr), np.array(R_arr), np.array(Lx_arr)
        ###############################################################################################################
        M_05k1 = M + 0.5*k1     
        R_05k1 = plmoOt20.calculate_radius_planet_Ot19(M_05k1)
        
        Mdot2 = mass_loss_rate_forward_Ot20(t+0.5*dt, epsilon, K_on, beta_on, planet_object, R_05k1, track_dict)
        #R_p=radius_planet(M + 0.5*k1))
        k2 = (dt*Myr_to_sec * Mdot2)/M_earth
        if np.iscomplex(k2):
            print("STOP this madness! Leaving volatile regime!")
            # if the stop criterium is reached, I add the core mass & core radius to the array at t_i+1
            print("t = ", t_arr[-1]+dt)
            t_arr.append(t_arr[-1]+dt)
            M_arr.append(plmoOt20.calculate_planet_mass_Ot20(R_core))
            R_arr.append(R_core)
            Lx_arr.append(Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))            
            return np.array(t_arr), np.array(M_arr), np.array(R_arr), np.array(Lx_arr)
        ###############################################################################################################
        M_05k2 = M + 0.5*k2
        R_05k2 = plmoOt20.calculate_radius_planet_Ot19(M_05k2)
        
        Mdot3 = mass_loss_rate_forward_Ot20(t+0.5*dt, epsilon, K_on, beta_on, planet_object, R_05k2, track_dict) 
        # R_p=radius_planet(M + 0.5*k2))
        k3 = (dt*Myr_to_sec * Mdot3)/M_earth
        if np.iscomplex(k3):
            print("STOP this madness! Leaving volatile regime!")
            # if the stop criterium is reached, I add the core mass & core radius to the array at t_i+1
            print("t = ", t_arr[-1]+dt)
            t_arr.append(t_arr[-1]+dt)
            M_arr.append(plmoOt20.calculate_planet_mass_Ot20(R_core))
            R_arr.append(R_core)
            Lx_arr.append(Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))            
            return np.array(t_arr), np.array(M_arr), np.array(R_arr), np.array(Lx_arr)
        ###############################################################################################################
        M_05k3 = M + 0.5*k3
        R_05k3 = plmoOt20.calculate_radius_planet_Ot19(M_05k3)
        
        Mdot4 = mass_loss_rate_forward_Ot20(t+dt, epsilon, K_on, beta_on, planet_object, R_05k3, track_dict) 
        # R_p=radius_planet(M + k3))
        k4 = (dt*Myr_to_sec * Mdot4)/M_earth
        if np.iscomplex(k4):
            print("STOP this madness! Leaving volatile regime!")
            # if the stop criterium is reached, I add the core mass & core radius to the array at t_i+1
            print("t = ", t_arr[-1]+dt)
            t_arr.append(t_arr[-1]+dt)
            M_arr.append(plmoOt20.calculate_planet_mass_Ot20(R_core))
            R_arr.append(R_core)
            Lx_arr.append(Lx_evo(t=t_arr[-1]+dt, track_dict=track_dict))            
            return np.array(t_arr), np.array(M_arr), np.array(R_arr), np.array(Lx_arr)
        ###############################################################################################################

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


##################################################################################################################################
##################################################################################################################################
##################################################################################################################################