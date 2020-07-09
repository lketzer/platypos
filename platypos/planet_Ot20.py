# file with my planet class
import os
import pandas as pd
import numpy as np

import astropy.units as u
from astropy import constants as const
from scipy.optimize import fsolve
import scipy.optimize as optimize

from platypos.lx_evo_and_flux import flux_at_planet, flux_at_planet_earth
from platypos.mass_evolution_function import mass_planet_RK4_forward_Ot20

class Planet_Ot20():
    """
    Structure of star_dictionary: {'star_id': "dummySun", 
                                   'mass': mass_star, 
                                   'radius': radius_star, 
                                   'age': age_star, 
                                   'L_bol': L_bol, 
                                   'Lx_age': Lx_age}
    Structure of planet_dict: {"radius": r_p, distance": a}
    """

    def __init__(self, star_dictionary, planet_dict, Lx_sat_info=None):
        '''Initialize a new instance (=object) of Planet_Ot20

        '''

        # initialize stellar parameters
        self.star_id = star_dictionary['star_id']
        self.mass_star = star_dictionary['mass']
        self.radius_star = star_dictionary['radius']
        self.age = star_dictionary['age']
        self.Lbol = star_dictionary['L_bol'] * const.L_sun.cgs.value
        self.Lx_age = star_dictionary['Lx_age']
        
        # initialize planet
        # -> based on observed radius, we estimate the mass from M-R relation
        # following params are the same for all planets!
        self.distance = planet_dict["distance"]
        self.flux = flux_at_planet_earth(self.Lbol, self.distance)
        self.has_evolved = False # flag which tracks if planet has been 
                                 # evolved and result file exists
        self.planet_id = "dummy" # set planet name later with set_name
        
        self.radius = planet_dict["radius"]
        self.mass_planet_Ot20()
        self.Lx_sat_info = Lx_sat_info

    def mass_planet_Ot20(self):
        """ calculate planet mass based on radius.
        I only use the volatile rich regime! This means the radius needs 
        to be greater than ~2.115 R_earth (radius with density > 3.3)
        """
        M_EARTH = const.M_earth.cgs.value
        R_EARTH = const.R_earth.cgs.value

        if (type(self.radius) == int) \
                or (type(self.radius) == float) \
                or (type(self.radius) == np.float64): # if R is single value
            M_p_volatile = 1.74 * (self.radius)**1.58 # if rho < 3.3 g/cm^3
            rho_volatile = M_p_volatile * M_EARTH \
                           / (4 / 3 * np.pi * (self.radius * R_EARTH)**3)
            if (rho_volatile >= 3.3):
                raise Exception("Planet with this radius is too small" \
                                + "and likely rocky; use LoFo14 models" \
                                + "instead.")
            else:
                if (M_p_volatile >= 120.):
                    raise Exception("Planet too massive. M-R relation only" \
                                    + "valid for <120 M_EARTH.")
                else:
                    self.mass = M_p_volatile

    def radius_planet_Ot20(self):
        """ calculate planet radius based on mass.
        I only use the volatile rich regime! This means the mass needs 
        to be bewteen ~5.7 and 120 M_earth
        """
        M_EARTH = const.M_earth.cgs.value
        R_EARTH = const.R_earth.cgs.value

        if (type(self.mass)==int) \
                or (type(self.mass) == float) \
                or (type(self.mass) == np.float64): # if M is single value
            R_p_volatile = (self.mass / 1.74)**(1. / 1.58)
            rho_volatile = self.mass * M_EARTH \
                           / (4 / 3 * np.pi * (R_p_volatile * R_EARTH)**3)
            if (rho_volatile >= 3.3):
                raise Exception("Planet with this mass/radius is too small" \
                                + "and likely rocky; use LoFo14 models" \
                                + "instead.")
            else:
                if (self.mass >= 120.):
                    raise Exception("Planet too massive. M-R relation only" \
                                    + "valid for <120 M_earth.")
                else:
                    self.radius = R_p_volatile

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
            t, M, R, Lx = mass_planet_RK4_forward_Ot20(
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
