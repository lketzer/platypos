# file with my planet class

import numpy as np
from scipy.optimize import fsolve
import astropy.units as u
from astropy import constants as const
import scipy.optimize as optimize
import os
import pandas as pd
from Lx_evo_and_flux import Lx_evo, flux_at_planet_earth, L_xuv_all, flux_at_planet
from Mass_evolution_function import mass_planet_RK4_forward_Ot14 # highest level function

class planet_Ot20():
    """
    Structure of star_dictionary: {'star_id': "dummySun", 'mass': mass_star, 'radius': radius_star, 
                                   'age': age_star, 'L_bol': L_bol, 'Lx_age': Lx_age}
    Structure of planet_dict: {"radius": r_p, distance": a} or 
    """
    def __init__(self, star_dictionary, planet_dict):
        
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
        self.has_evolved = False # flag which tracks if planet has been evolved and result file exists
        self.planet_id = "None"
        
        self.radius = planet_dict["radius"]
        self.mass_planet_Ot19()
#         while True:
#             try:
#                 self.radius = planet_dict["radius"]
#                 mass_planet_Ot19(self.radius)
          
    # Class Methods
    
    def mass_planet_Ot19(self):
        """
        I only use the volatile rich regime! This means the radius needs 
        to be greater than ~2.115 R_earth (radius with density > 3.3)
        """
        M_earth = const.M_earth.cgs.value
        R_earth = const.R_earth.cgs.value

        if (type(self.radius)==int) or (type(self.radius)==float) or (type(self.radius)==np.float64): # if R is single value
            M_p_volatile = 1.74*(self.radius)**1.58 # if rho < 3.3 g/cm^3
            rho_volatile = M_p_volatile*M_earth/(4/3*np.pi*(self.radius*R_earth)**3)
            if (rho_volatile >= 3.3):
                raise Exception("Planet with this radius is too small and likely rocky; use LoFo14 models instead.")
            else:
                if (M_p_volatile >= 120):
                    raise Exception("Planet too massive. M-R relation only valid for <120 M_earth.")
                else:
                    self.mass = M_p_volatile

    def radius_planet_Ot19(self):
        """
        I only use the volatile rich regime! This means the mass needs 
        to be bewteen ~5.7 and 120 M_earth
        """
        M_earth = const.M_earth.cgs.value
        R_earth = const.R_earth.cgs.value

        if (type(self.mass)==int) or (type(self.mass)==float) or (type(self.mass)==np.float64): # if M is single value
            #R_p_volatile = 0.7*self.mass**0.63 # if rho < 3.3 g/cm^3
            R_p_volatile = (self.mass/1.74)**(1./1.58)
            rho_volatile = self.mass*M_earth/(4/3*np.pi*(R_p_volatile*R_earth)**3)
            if (rho_volatile >= 3.3):
                raise Exception("Planet with this mass/radius is too small and \
                                likely rocky; use LoFo14 models instead.")
            else:
                if (self.mass >= 120):
                    raise Exception("Planet too massive. M-R relation only valid for <120 M_earth.")
                else:
                    self.radius = R_p_volatile
        
    def set_name(self, t_final, initial_step_size, epsilon, K_on, beta_on, evo_track_dict):
        """ function to set the right planet name based on the track specified. This can then be used to check if 
        a particular planet on a particular track has already evolved. """
        self.planet_id = 'planet_a'+str(np.round(self.distance, 3)).replace('.', 'dot') + \
                         '_Mcore'+str(np.round(self.core_mass,3)).replace(".", "dot") + '_fenv' + \
                         str(np.round(self.fenv,3)) + '_' + self.metallicity + \
                         '_Mstar' + str(np.round(self.mass_star,3)).replace(".", "dot") + "_K_" + K_on + \
                         "_beta_" + beta_on + \
                         "_track_" + str(evo_track_dict["t_start"]) + "_" + str(evo_track_dict["t_sat"]) + \
                         "_" + str(t_final) + "_" + str(evo_track_dict["Lx_max"]) + \
                         "_" + str(evo_track_dict["dt_drop"]) + "_" + str(evo_track_dict["Lx_drop_factor"])
        
    
    def evolve_forward(self, t_final, initial_step_size, epsilon, K_on, beta_on, evo_track_dict, path_for_saving, planet_folder_id):
        """ Call this function to make the planet evolve. """
        # create a planet ID -> used for filname to save the result
#         self.planet_id = 'planet_a'+str(np.round(self.distance, 3)).replace('.', 'dot') + \
#                          '_Mcore'+str(np.round(self.core_mass,3)).replace(".", "dot") + '_fenv' + \
#                          str(np.round(self.fenv,3)) + '_' + self.metallicity + \
#                          '_Mstar' + str(np.round(self.mass_star,3)).replace(".", "dot") + "_K_" + K_on + "_beta_" + beta_on + \
#                          "_track_" + str(evo_track_dict["t_start"]) + "_" + str(evo_track_dict["t_sat"]) + \
#                          "_" + str(t_final) + "_" + str(evo_track_dict["Lx_max"]) + \
#                         "_" + str(evo_track_dict["dt_drop"]) + "_" + str(evo_track_dict["Lx_drop_factor"])
        
        # write info with planet parameters to file!!
        # also track params!
        
        self.planet_id = planet_folder_id + "_track_" + str(evo_track_dict["t_start"]) + "_" + str(evo_track_dict["t_sat"]) + \
                         "_" + str(t_final) + "_" + str(evo_track_dict["Lx_max"]) + \
                        "_" + str(evo_track_dict["dt_drop"]) + "_" + str(evo_track_dict["Lx_drop_factor"])
        
        if os.path.exists(path_for_saving+self.planet_id+".txt"):
            #print("Planet already existis!")
            self.has_evolved = True
        else:
            print("Planet: ", self.planet_id+".txt")
            
            ###################################
            if os.path.exists(path_for_saving+planet_folder_id+".txt"):
                pass
            else:
                # create a file: planet_XXXX.txt which contains the initial planet
                p = open(path_for_saving+planet_folder_id+".txt", "a")
                planet_params = "a,mass,radius\n"+ str(self.distance) + "," + str(self.mass) + "," \
                                                                   + str(self.radius)
                p.write(planet_params)
                p.close()
                
            # create a file: planet_XXXX.txt which contains the track params
            t = open(path_for_saving+"track_params_"+self.planet_id+".txt", "a") 
            track_params = "t_start,t_sat,t_curr,t_5Gyr,Lx_max,Lx_curr,Lx_5Gyr,dt_drop,Lx_drop_factor\n" \
                            + str(evo_track_dict["t_start"]) + "," + str(evo_track_dict["t_sat"]) + "," \
                            + str(evo_track_dict["t_curr"]) + "," + str(evo_track_dict["t_5Gyr"]) + "," \
                            + str(evo_track_dict["Lx_max"]) + "," + str(evo_track_dict["Lx_curr"]) + "," \
                            + str(evo_track_dict["Lx_5Gyr"]) + "," + str(evo_track_dict["dt_drop"]) + "," \
                            + str(evo_track_dict["Lx_drop_factor"])
            t.write(track_params)
            t.close()
            
            if os.path.exists(path_for_saving+"host_star_properties.txt"):
                pass
            else:
                # create a file: host_star_properties.txt which contains the host star params
                s = open(path_for_saving+"host_star_properties.txt", "a") 
                star_params = "star_id,mass_star,radius_star,age,Lbol,Lx_age\n" + self.star_id  + "," + \
                                str(self.mass_star)  + "," + str(self.radius_star) + "," +  str(self.age) + "," +  \
                                str(self.Lbol/const.L_sun.cgs.value) + "," + str(self.Lx_age)
                s.write(star_params)
                s.close()
            ###################################
            
            print("Start evolving.")
            t, M, R, Lx = mass_planet_RK4_forward_Ot14(epsilon=epsilon, K_on="yes", beta_on="yes", planet_object=self, 
                                                       initial_step_size=initial_step_size, t_final=t_final, 
                                                       track_dict=evo_track_dict)
            df = pd.DataFrame({"Time": t, "Mass": M, "Radius": R, "Lx": Lx})
            df.to_csv(path_for_saving+self.planet_id+".txt", index=None)
            print("Saved!")
            self.has_evolved = True
    
    def read_results(self, file_path):
        if self.has_evolved == True: # planet exists, has been evolved & results (which are stored in file_path) can be read in
            df = pd.read_csv(file_path+self.planet_id+".txt")
            t, M, R, Lx = df["Time"].values, df["Mass"].values, df["Radius"].values, df["Lx"].values
            return t, M, R, Lx
        else:
            print("Planet has not been evolved & no result file exists.")
    
    
            
    def evolve_backwards(self, t_final, initial_step_size, epsilon, K_on, beta_on, evo_track_dict, path_for_saving):
        contine