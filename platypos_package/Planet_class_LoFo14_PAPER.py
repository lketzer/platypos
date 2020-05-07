# file with my planet class

import numpy as np
from scipy.optimize import fsolve
import astropy.units as u
from astropy import constants as const
import scipy.optimize as optimize
import os
import pandas as pd
from Lx_evo_and_flux import Lx_evo, flux_at_planet_earth, L_xuv_all, flux_at_planet
from Mass_evolution_function import mass_planet_RK4_forward_LO14_PAPER # highest level function

class planet_LoFo14_PAPER():
    """
    Need star and planet dictionary to initialize a planet object.
    Structure of star_dictionary: {'star_id': "dummySun", 'mass': mass_star, 'radius': radius_star, 
                                   'age': age_star, 'L_bol': L_bol, 'Lx_age': Lx_age}
    Structure of planet_dict: {"core_mass": m_c, "fenv": f, "distance": a, "metallicity": metal}
    """
    def __init__(self, star_dictionary, planet_dict):
        
        # initialize stellar parameters
        self.star_id = star_dictionary['star_id']
        self.mass_star = star_dictionary['mass']
        self.radius_star = star_dictionary['radius']
        self.age = star_dictionary['age']
        self.Lbol = star_dictionary['L_bol'] * const.L_sun.cgs.value
        self.Lx_age = star_dictionary['Lx_age']
        
        # initialize planet parameters
        # I have three options for planet_dict:
                # 1) artificial planet with M_core & f_env -> need to calculate M_pl & R_pl
                # 2) observed planet with a mass: M_pl & R_pl & f_env -> M_core 
                # 3) observed planet w/o mass: M_core, R_pl -> calulate f_env & M_pl
        
        # following 3 params are the same for all planets!
        self.distance = planet_dict["distance"]
        self.metallicity = planet_dict["metallicity"]  # solarZ or enhZ
        self.flux = flux_at_planet_earth(self.Lbol, self.distance)
        self.has_evolved = False # flag which tracks if planet has been evolved and result file exists
        self.planet_id = "None"
        
        # the remaining parameters depend on if you choose a case 1, 2, or 3 planet
        while True:
            try:
                planet_dict["fenv"]
                #print("Case 1 - artificial planet (with M_core & f_env)") # Case 1
                self.planet_info = "Case 1 - artificial planet (with M_core & f_env)"
                # Case 1: artificial planet with fenv & M_core given -> need to calculate the planet mass
                self.fenv = planet_dict["fenv"]
                self.core_mass = planet_dict["core_mass"]
                self.Calculate_core_radius()
                self.Calculate_planet_mass()
                self.Calculate_planet_radius()
                break
            except KeyError: # Case 2 or 3
                #print("No fenv given.")
                while True:
                    try:
                        planet_dict["mass"]
                        # Case 3 - observed planet with radius but no mass & M_core specified -> need to calculate/estimate fenv & M_pl
                        #print("Case 3 - obs. planet with radius & mass measurement (with R_pl, M_pl, M_core)")
                        self.planet_info = "Case 3 - obs. planet with radius & mass measurement (with R_pl, M_pl, M_core)"
                        self.core_mass = planet_dict["core_mass"]
                        self.mass = planet_dict["mass"]
                        self.radius = planet_dict["radius"]
                        self.Calculate_core_radius()
                        self.Solve_for_fenv() # function to solve for the envelope mass fraction
                        # add sanity check to make sure the mass with the calculated fenv matches the observed input mass!
                        # add sanity check to make sure the radius with the calculated fenv matches the observed input radius!
                        
                        break
                    except KeyError:
                        #print("Case 2 - obs. planet with radius, but no mass (with R_pl, M_core)") 
                        self.planet_info = "Case 2 - obs. planet with radius, but no mass (with R_pl, M_core)"
                        # Case 2 - observed planet with a mass, radius & core mass specified -> need to calculate fenv
                        self.core_mass = planet_dict["core_mass"]
                        self.radius = planet_dict["radius"]
                        self.Calculate_core_radius()
                        self.Solve_for_fenv() # function to solve for the envelope mass fraction
                        self.Calculate_planet_mass()
                        # add sanity check to make sure the radius with the calculated fenv matches the observed input radius!
                        
                        break
                break

    # Class Methods
    def Calculate_planet_mass(self):
        """ Planet mass determined by core mass and atmosphere mass (specified in terms of atm. mass fraction [%]). """
        #M_core = (self.core_mass/const.M_earth.cgs).value
        self.mass = self.core_mass/(1-(self.fenv/100))
    
    def Calculate_planet_core_mass(self):
        """ Planet core mass determined by planet mass and atmosphere mass (specified in terms of atm. mass fraction [%]). """
        self.core_mass = self.mass*(1-(self.fenv/100))
     
    def Calculate_core_radius(self):
        """M-R relation for rock/iron Earth-like core. (no envelope)"""
        self.core_radius = (self.core_mass**0.25)

    def Solve_for_fenv(self):
        if self.radius == self.core_radius:
            self.fenv = 0.0
        else:
            def Calculate_fenv(fenv):
                age_exponent = {"solarZ": -0.11, "enhZ": -0.18}  
                return -self.radius + self.core_radius + (2.06 * (self.core_mass/(1-(fenv/100)))**(-0.21) \
                                                        * (fenv/5)**0.59 * (self.flux)**0.044 * \
                                                          ((self.age/1e3)/5)**(age_exponent[self.metallicity]))
            f_guess = 0.1
            fenv = optimize.fsolve(Calculate_fenv, x0=f_guess)[0]
            #print("fenv = ", fenv)
            self.fenv = fenv
            #if fenv1 != fenv:
            #    print("Sth went wrong in solving for the envelope mass fraction! Check!")

        
    def Calculate_R_env(self):
        """ M_p in units of Earth masses, f_env in percent, Flux in earth units, age in Gyr
        R_env ~ t**0.18 for *enhanced opacities*;
        R_env ~ t**0.11 for *solar metallicities* 
        """
        age_exponent = {"solarZ": -0.11, "enhZ": -0.18}   
        R_env = 2.06 * self.mass**(-0.21) * (self.fenv/5)**0.59 * \
                self.flux**0.044 * ((self.age/1e3)/5)**(age_exponent[self.metallicity]) 
        return R_env # in units of R_earth


    def Calculate_planet_radius(self):
        """ description: """
        self.radius = self.core_radius + self.Calculate_R_env()
    
    def set_name(self, t_final, initial_step_size, epsilon, K_on, beta_on, evo_track_dict):
        """ function to set the right planet name based on the track specified. This can then be used to check if 
        a particular planet on a particular track has already evolved. """
        self.planet_id = 'planet_a'+str(np.round(self.distance, 3)).replace('.', 'dot') + \
                         '_Mcore'+str(np.round(self.core_mass,3)).replace(".", "dot") + '_fenv' + \
                         str(np.round(self.fenv,3)) + '_' + self.metallicity + \
                         '_Mstar' + str(np.round(self.mass_star,3)).replace(".", "dot") + "_K_" + K_on + "_beta_" + beta_on + \
                         "_track_" + str(evo_track_dict["t_start"]) + "_" + str(evo_track_dict["t_sat"]) + \
                         "_" + str(t_final) + "_" + str(evo_track_dict["Lx_max"]) + \
                        "_" + str(evo_track_dict["dt_drop"]) + "_" + str(evo_track_dict["Lx_drop_factor"])
        
    
    def evolve_forward(self, t_final, initial_step_size, epsilon, K_on, beta_on, evo_track_dict, path_for_saving, planet_folder_id):
        """ Call this function to make the planet evolve. """
        
        # set planet id
        self.planet_id = planet_folder_id + "_track_" + str(evo_track_dict["t_start"]) + "_" + str(evo_track_dict["t_sat"]) + \
                         "_" + str(t_final) + "_" + str(evo_track_dict["Lx_max"]) + \
                        "_" + str(evo_track_dict["dt_drop"]) + "_" + str(evo_track_dict["Lx_drop_factor"])
        
        if os.path.exists(path_for_saving+self.planet_id+".txt"):
            #print("Planet already existis!")
            self.has_evolved = True
        else:
            print("Planet: ", self.planet_id+".txt")
            
            ###################################
            # create a file: planet_XXXX.txt which contains the initial planet params
            if os.path.exists(path_for_saving+planet_folder_id+".txt"):
                pass
            else:
                p = open(path_for_saving+planet_folder_id+".txt", "a") 
                planet_params = "a,core_mass,fenv,mass,radius,metallicity,age\n" + \
                                str(self.distance) + "," + str(self.core_mass) + "," + \
                                str(self.fenv) + "," + str(self.mass) + "," + str(self.radius) + "," + \
                                self.metallicity + "," + str(self.age)
                p.write(planet_params)
                p.close() 
            ###################################
            # create a file: planet_XXXX.txt which contains the track params
            if os.path.exists(path_for_saving+"track_params_"+self.planet_id+".txt"):
                pass
            else:
                t = open(path_for_saving+"track_params_"+self.planet_id+".txt", "a") 
                track_params = "t_start,t_sat,t_curr,t_5Gyr,Lx_max,Lx_curr,Lx_5Gyr,dt_drop,Lx_drop_factor\n" \
                                + str(evo_track_dict["t_start"]) + "," + str(evo_track_dict["t_sat"]) + "," \
                                + str(evo_track_dict["t_curr"]) + "," + str(evo_track_dict["t_5Gyr"]) + "," \
                                + str(evo_track_dict["Lx_max"]) + "," + str(evo_track_dict["Lx_curr"]) + "," \
                                + str(evo_track_dict["Lx_5Gyr"]) + "," + str(evo_track_dict["dt_drop"]) + "," \
                                + str(evo_track_dict["Lx_drop_factor"])
                t.write(track_params)
                t.close()
            ###################################
            # create a file: host_star_properties.txt which contains the host star params
            if os.path.exists(path_for_saving+"host_star_properties.txt"):
                pass
            else:
                s = open(path_for_saving+"host_star_properties.txt", "a") 
                star_params = "star_id,mass_star,radius_star,age,Lbol,Lx_age\n" + self.star_id  + "," + \
                                str(self.mass_star)  + "," + str(self.radius_star) + "," +  str(self.age) + "," +  \
                                str(self.Lbol/const.L_sun.cgs.value) + "," + str(self.Lx_age)
                s.write(star_params)
                s.close()
            ###################################
        
            print("Start evolving.")
            t, M, R, Lx = mass_planet_RK4_forward_LO14_PAPER(epsilon=epsilon, K_on="yes", beta_on="yes", planet_object=self, 
                                                           initial_step_size=initial_step_size, t_final=t_final, 
                                                           track_dict=evo_track_dict)
            df = pd.DataFrame({"Time": t, "Mass": M, "Radius": R, "Lx": Lx})
            df.to_csv(path_for_saving+self.planet_id+".txt", index=None)
            print("Saved!")
            
            # make another file, which contains the final parameters
            if os.path.exists(path_for_saving+planet_folder_id+"_"+self.planet_id+"_final.txt"):
                pass
            else:
                p = open(path_for_saving+self.planet_id+"_final.txt", "a") 
                index_of_last_entry = df["Radius"][df["Radius"].notna()].index[-1]
                R_final = df["Radius"].loc[index_of_last_entry]
                index_of_last_entry = df["Mass"][df["Mass"].notna()].index[-1]
                M_final = df["Mass"].loc[index_of_last_entry]
                planet_params = "a,core_mass,mass,radius,metallicity,track\n" + \
                                str(self.distance) + "," + str(self.core_mass) + "," + \
                                str(M_final) + "," + str(R_final) + "," + self.metallicity + "," + \
                                self.planet_id
                p.write(planet_params)
                p.close()    
            
            self.has_evolved = True
    
    def read_results(self, file_path):
        if self.has_evolved == True: # planet exists, has been evolved & results (which are stored in file_path) can be read in
            df = pd.read_csv(file_path+self.planet_id+".txt", float_precision='round_trip') 
            # to avoid pandas doing sth. weird to the last digit
            # Pandas uses a dedicated dec 2 bin converter that compromises accuracy in preference to speed.
            # Passing float_precision='round_trip' to read_csv fixes this.
            t, M, R, Lx = df["Time"].values, df["Mass"].values, df["Radius"].values, df["Lx"].values
            return t, M, R, Lx
        else:
            print("Planet has not been evolved & no result file exists.")
    
    
            
    def evolve_backwards(self, t_final, initial_step_size, epsilon, K_on, beta_on, evo_track_dict, path_for_saving):
        contine