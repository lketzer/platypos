# file with my planet class
import os
import numpy as np
import pandas as pd

import astropy.units as u
from astropy import constants as const
import scipy.optimize as optimize
from scipy.optimize import fsolve

from platypos.lx_evo_and_flux import flux_at_planet, flux_at_planet_earth
from platypos.mass_evolution_function import mass_planet_RK4_forward_LO14_PAPER


class Planet_LoFo14_PAPER():
    """
    Need star and planet dictionary to initialize a planet object.
    Structure of star_dictionary: {'star_id': "dummySun", 'mass': mass_star, 'radius': radius_star, 
                                   'age': age_star, 'L_bol': L_bol, 'Lx_age': Lx_age}
    Structure of planet_dict: {"core_mass": m_c, "fenv": f, "distance": a, "metallicity": metal}
    """
    def __init__(self, star_dictionary, planet_dict, Lx_sat_info=None):
        
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
        self.Lx_sat_info = Lx_sat_info
        
        # the remaining parameters depend on if you choose a case 1, 2, or 3 planet
        while True:
            try:
                planet_dict["fenv"]
                #print("Case 1 - artificial planet (with M_core & f_env)") # Case 1
                self.planet_info = "Case 1 - artificial planet (with M_core & f_env)"
                # Case 1: artificial planet with fenv & M_core given -> need to calculate the planet mass
                self.fenv = planet_dict["fenv"]
                self.core_mass = planet_dict["core_mass"]
                self.calculate_core_radius()
                self.calculate_planet_mass()
                self.calculate_planet_radius()
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
                        self.calculate_core_radius()
                        self.solve_for_fenv() # function to solve for the envelope mass fraction
                        # add sanity check to make sure the mass with the calculated fenv matches the observed input mass!
                        # add sanity check to make sure the radius with the calculated fenv matches the observed input radius!
                        
                        break
                    except KeyError:
                        #print("Case 2 - obs. planet with radius, but no mass (with R_pl, M_core)") 
                        self.planet_info = "Case 2 - obs. planet with radius, but no mass (with R_pl, M_core)"
                        # Case 2 - observed planet with a mass, radius & core mass specified -> need to calculate fenv
                        self.core_mass = planet_dict["core_mass"]
                        self.radius = planet_dict["radius"]
                        self.calculate_core_radius()
                        self.solve_for_fenv() # function to solve for the envelope mass fraction
                        self.calculate_planet_mass()
                        # add sanity check to make sure the radius with the calculated fenv matches the observed input radius!
                        
                        break
                break

    # Class Methods
    def calculate_planet_mass(self):
        """ Planet mass determined by core mass and atmosphere mass (specified in terms of atm. mass fraction [%]). """
        #M_core = (self.core_mass/const.M_earth.cgs).value
        self.mass = self.core_mass/(1-(self.fenv/100))
    
    def calculate_planet_core_mass(self):
        """ Planet core mass determined by planet mass and atmosphere mass (specified in terms of atm. mass fraction [%]). """
        self.core_mass = self.mass*(1-(self.fenv/100))
     
    def calculate_core_radius(self):
        """M-R relation for rock/iron Earth-like core. (no envelope)"""
        self.core_radius = (self.core_mass**0.25)

    def solve_for_fenv(self):
        if self.radius == self.core_radius:
            self.fenv = 0.0
        else:
            def calculate_fenv(fenv):
                age_exponent = {"solarZ": -0.11, "enhZ": -0.18}  
                return -self.radius + self.core_radius + (2.06 * (self.core_mass/(1-(fenv/100)))**(-0.21) \
                                                        * (fenv/5)**0.59 * (self.flux)**0.044 * \
                                                          ((self.age/1e3)/5)**(age_exponent[self.metallicity]))
            f_guess = 0.1
            fenv = optimize.fsolve(calculate_fenv, x0=f_guess)[0]
            #print("fenv = ", fenv)
            self.fenv = fenv
            #if fenv1 != fenv:
            #    print("Sth went wrong in solving for the envelope mass fraction! Check!")

        
    def calculate_R_env(self):
        """ M_p in units of Earth masses, f_env in percent, Flux in earth units, age in Gyr
        R_env ~ t**0.18 for *enhanced opacities*;
        R_env ~ t**0.11 for *solar metallicities* 
        """
        age_exponent = {"solarZ": -0.11, "enhZ": -0.18}   
        R_env = 2.06 * self.mass**(-0.21) * (self.fenv/5)**0.59 * \
                self.flux**0.044 * ((self.age/1e3)/5)**(age_exponent[self.metallicity]) 
        return R_env # in units of R_earth


    def calculate_planet_radius(self):
        """ description: """
        self.radius = self.core_radius + self.calculate_R_env()
    
    def set_name(self, t_final, initial_step_size, 
                 epsilon, K_on, beta_on, evo_track_dict):
        """ function to set the right planet name based on the 
        track specified. This can then be used to check if a particular
        planet on a particular track has already evolved. 
        """
        self.planet_id = "planet_" \
                         + str(np.round(self.distance, 3)).replace('.', 'dot') \
                         + 'radius' \
                         + str(np.round(self.radius,3)).replace(".", "dot") \
                         + '_mass' + str(np.round(self.mass,3)) \
                         + '_Mstar' \
                         + str(np.round(self.mass_star,3)).replace(".", "dot") \
                         + "_K_" + K_on + "_beta_" + beta_on \
                         + "_track_" + str(evo_track_dict["t_start"]) + "_" \
                         + str(evo_track_dict["t_sat"]) + "_" \
                         + str(t_final) + "_" + str(evo_track_dict["Lx_max"]) \
                         + "_" + str(evo_track_dict["dt_drop"]) \
                         + "_" + str(evo_track_dict["Lx_drop_factor"])

    def generate_planet_id(self, t_final, 
                           planet_folder_id, evo_track_dict):
        """ Similar as set_name; produces a planet_id which can be used 
            for saving the results. """
        self.planet_id = planet_folder_id + "_track_" \
                         + str(evo_track_dict["t_start"]) + "_" \
                         + str(evo_track_dict["t_sat"]) + "_" \
                         + str(t_final) + "_" + str(evo_track_dict["Lx_max"]) \
                         + "_" + str(evo_track_dict["dt_drop"]) + "_" \
                         + str(evo_track_dict["Lx_drop_factor"])
    

    def write_general_params_to_file(self, path_for_saving, 
                                     planet_folder_id, evo_track_dict):
        """produces various files:
            - a file with the initial planet parameters (name is the
              planet folder name, e.g. planet_001.txt), 
            - a file with the host star parameters (host_star.txt), 
            - a file with the corresponding track parameters starting with 
              "track_params_"
        """ 
        # create a file (e.g. planet_XXXX.txt) which contains the
        # initial planet params
        if not os.path.exists(path_for_saving + planet_folder_id + ".txt"):
            with open(path_for_saving + planet_folder_id + ".txt", "w") as p:
                planet_params = "a,mass,radius,age\n" \
                                 + str(self.distance) + "," \
                                 + str(self.mass) + "," \
                                 + str(self.radius) + "," \
                                 + str(self.age)
                p.write(planet_params)

        # create a file (track_params_planet_....txt) which contains
        # the track parameters
        if not os.path.exists(path_for_saving + "track_params_" \
                              + self.planet_id + ".txt"):
            with open(path_for_saving + "track_params_" + self.planet_id + ".txt", "w") as t:
                track_params = "t_start,t_sat,t_curr,t_5Gyr,Lx_max,Lx_curr," \
                                + "Lx_5Gyr,dt_drop,Lx_drop_factor\n" \
                                + str(evo_track_dict["t_start"]) + "," \
                                + str(evo_track_dict["t_sat"]) + ","  \
                                + str(evo_track_dict["t_curr"]) + "," \
                                + str(evo_track_dict["t_5Gyr"]) + "," \
                                + str(evo_track_dict["Lx_max"]) + "," \
                                + str(evo_track_dict["Lx_curr"]) + "," \
                                + str(evo_track_dict["Lx_5Gyr"]) + "," \
                                + str(evo_track_dict["dt_drop"]) + "," \
                                + str(evo_track_dict["Lx_drop_factor"])
                t.write(track_params)

        # create a file which contains the host star parameters
        if not os.path.exists(path_for_saving + "host_star_properties.txt"):
            with open(path_for_saving + "host_star_properties.txt", "w") as s:
                star_params = "star_id,mass_star,radius_star,age,Lbol,Lx_age\n" \
                              + self.star_id + "," + str(self.mass_star) \
                              + "," + str(self.radius_star) \
                              + "," + str(self.age) \
                              + "," + str(self.Lbol / const.L_sun.cgs.value) \
                              + "," + str(self.Lx_age)
                s.write(star_params)

    def write_final_params_to_file(self, results_df, path_for_saving):
        """ Create file with only the final time, mass and radius parameters. """ 
        
        # create another file, which contains the final parameters only
        if not os.path.exists(path_for_saving \
                              + self.planet_id + "_final.txt"):
            with open(path_for_saving + self.planet_id + "_final.txt", "w") as p:
                # get last element (final time, mass, radius)
                t_final = results_df["Time"].iloc[-1]
                R_final = results_df["Radius"].iloc[-1]
                M_final = results_df["Mass"].iloc[-1]
                planet_params = "a,time,mass,radius,track\n" \
                                + str(self.distance) + "," \
                                + str(t_final) + "," \
                                + str(M_final) + "," \
                                + str(R_final) + "," \
                                + self.planet_id
                p.write(planet_params)

    def evolve_forward(self, t_final,
                       initial_step_size,
                       epsilon, K_on, beta_on,
                       evo_track_dict, 
                       path_for_saving,
                       planet_folder_id):
        """ Call this function to make the planet evolve and 
        create file with mass and radius evolution. 
        See Mass_evolution_function.py for details on the integration.
        """
        
        if os.path.exists(path_for_saving + self.planet_id + ".txt"):
            # planet already exists
            self.has_evolved = True
            df = pd.read_csv(path_for_saving + self.planet_id + ".txt")
        else:
            #print("Planet: ", self.planet_id+".txt")
            # call mass_planet_RK4_forward_LO14 to start the integration
            t, M, R, Lx = mass_planet_RK4_forward_LO14_PAPER(
                                    epsilon=epsilon,
                                    K_on=K_on,
                                    beta_on=beta_on,
                                    planet_object=self,
                                    initial_step_size=initial_step_size,
                                    t_final=t_final,
                                    track_dict=evo_track_dict
                                    )
            
            ### TO DO: move this to mass_planet_RK4_forward_LO14 -
            # > make it return dataframe
            # add results to dataframe and save
            df = pd.DataFrame({"Time": t, "Mass": M, "Radius": R, "Lx": Lx})
            df.to_csv(path_for_saving + self.planet_id + ".txt", index=None)
            self.has_evolved = True  # set evolved-flag to True
            
        return df

    def evolve_forward_and_create_full_output(self, t_final,
                                              initial_step_size,
                                              epsilon,
                                              K_on,
                                              beta_on,
                                              evo_track_dict,
                                              path_for_saving,
                                              planet_folder_id
                                              ):
        """ This is the master-function which needs to be called to 
        evolve the planet and at the same time create all necessary 
        output files which contain useful data about initial params, 
        host star params, tracks etc..."""
        
        # create planet id for file names
        self.generate_planet_id(t_final, planet_folder_id, evo_track_dict)
        # create file with initial planet, host star and track params
        self.write_general_params_to_file(path_for_saving,
                                          planet_folder_id,
                                          evo_track_dict)
        
        results_df = self.evolve_forward(t_final, initial_step_size,
                                         epsilon, K_on, beta_on,
                                         evo_track_dict,
                                         path_for_saving,
                                         planet_folder_id)
        
        # create file with final planet params
        self.write_final_params_to_file(results_df, path_for_saving)
        
    def read_results(self, file_path):
        """  read in results file and return dataframe. """
        # planet exists, has been evolved and results can be read in
        if self.has_evolved == True:
            df = pd.read_csv(file_path+self.planet_id+".txt",
                             float_precision='round_trip')
            # 'round_trip': to avoid pandas doing sth. weird to the last digit
            # Info: Pandas uses a dedicated dec 2 bin converter that
            # a compromisesccuracy in preference to speed.
            # Passing float_precision='round_trip' to read_csv fixes this.
            #t, M, R, Lx = df["Time"].values, df["Mass"].values,\
            #              df["Radius"].values, df["Lx"].values
            return df #t, M, R, Lx
        else:
            print("Planet has not been evolved & no result file exists.")
              
    def evolve_backwards(self):
        raise NotImplementedError("Coming soon! :)")