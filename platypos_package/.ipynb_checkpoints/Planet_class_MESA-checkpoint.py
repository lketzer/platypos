# file with my planet class

import numpy as np
from scipy.optimize import fsolve
import astropy.units as u
from astropy import constants as const
import scipy.optimize as optimize
from scipy.interpolate import griddata
import os
import pandas as pd
from Lx_evo_and_flux import Lx_evo, flux_at_planet_earth, L_xuv_all, flux_at_planet
from Mass_evolution_function import mass_planet_RK4_forward_MESA # highest level function
import Planet_model_MESA as plmoMESA

class planet_MESA():
    """
    Structure of star_dictionary: {'star_id': "dummySun", 'mass': mass_star, 'radius': radius_star, 
                                   'age': age_star, 'L_bol': L_bol, 'Lx_age': Lx_age}
    Planet needs age: in star dictionary!
    Structure of planet_dict: {"mass": r_p, "core_mass": mcore, distance": a} (need to calculate mass) 
    or
    #Structure of planet_dict: {"core_mass": mcore, "distance": a} (need to calculate radius) 
    NOTE: mass & radius should be in units of Earth
    """
    def __init__(self, star_dictionary, planet_dict, interp_R_dict):
        
        # initialize the interpolation (done once)
        #interp_R_dict = plmoMESA.initialize_R_interpolation_MESA_grid()
        # 2D interpolation functions from plmoMESA.initialize_R_interpolation_MESA_grid()
        self.R_interpolation_functions = interp_R_dict
        #interp_M_dict
        
        # initialize stellar parameters
        self.star_id = star_dictionary['star_id']
        self.mass_star = star_dictionary['mass']
        self.radius_star = star_dictionary['radius']
        self.age = star_dictionary['age']
        self.Lbol = star_dictionary['L_bol'] * const.L_sun.cgs.value
        self.Lx_age = star_dictionary['Lx_age']
        
        # initialize planet
        # -> based on (observed radius), core_mass, a and age, we estimate the mass from MESA models
        
        # following params are the same for all planets!
        self.distance = planet_dict["distance"]
        self.flux = flux_at_planet_earth(self.Lbol, self.distance)
        self.has_evolved = False # flag which tracks if planet has been evolved and result file exists
        self.planet_id = "None"
        #self.grid = planet_dict["grid"]
        
        try:
            self.mass = planet_dict["mass"]
            self.core_mass = planet_dict["core_mass"]
            self.mass_jup = self.mass * (const.M_earth/const.M_jup).value
            self.radius_planet_MESA()
        except KeyError:
            # having a given radius instead of mass is not so good!
            self.radius = planet_dict["radius"]
            self.core_mass = planet_dict["core_mass"]
            self.radius_jup = self.radius * (const.R_earth/const.R_jup).value
            self.mass_planet_MESA()
          
    # Class Methods
    
    def radius_planet_MESA(self):
        """ 
        M_pl: input in Earth masses but grid in M_jup 
        Use MESA models to calculate planetary mass at a given radius, orbital separation and age
        NOTE: not all R_pl, a, age combinations might have a solution!
        NOTE: for now all planets have a 10 M_earth core with density 8 g/cm^3
        """
        R = self.R_interpolation_functions["Mcore"+str(int(float(self.core_mass)))][str(self.distance)](self.age, self.mass_jup)[0][0]
        self.radius_jup = R
        self.radius = R*(const.R_jup/const.R_earth).value
    
    def mass_planet_MESA(self):
        """
        USE with CAUTION! same radius-age combination can have low or high mass, so this might not work.
        Use MESA models to calculate planetary mass at a given radius, orbital separation and age
        NOTE: not all R_pl, a, age combinations might have a solution!
        NOTE: for now all planets have a 10 M_earth core with density 8 g/cm^3
        """
        # this step takes some time; USE WISELY
        ####################################################################
        # data to use for calculating the planetary radius in PLATYPOS
        pathdir = os.getcwd().split("gitlab")[0]+'gitlab/mesagiants//Results_population4_2/Tables/'
        grid_Mcore0 = "PLATYPOS_poulation4_2_diffRstart_M_R_age_orbsep_5_to_6000Myr_Mcore0_origM_dense.csv"
        grid_Mcore10 = "PLATYPOS_poulation4_2_diffRstart_M_R_age_orbsep_5_to_6000Myr_Mcore10_origM_dense.csv"
        grid_Mcore25 = "PLATYPOS_poulation4_2_diffRstart_M_R_age_orbsep_5_to_6000Myr_Mcore25_origM_dense.csv"
        dfPLATYPOS_Mcore0 = pd.read_csv(pathdir + grid_Mcore0)
        dfPLATYPOS_Mcore10 = pd.read_csv(pathdir + grid_Mcore10)
        dfPLATYPOS_Mcore25 = pd.read_csv(pathdir + grid_Mcore25)
        dict_dfPLATYPOS = {"Mcore0": dfPLATYPOS_Mcore0, 
                           "Mcore10": dfPLATYPOS_Mcore10, 
                           "Mcore25": dfPLATYPOS_Mcore25}
        # read in the data once, then only work with it
        # grid of known radii and distances & R
        # for all ages
        a_R_age_points = []
        M_age_values = []
        
        dfPLATYPOS = dict_dfPLATYPOS["Mcore"+str(int(float(mass_core)))]
        for age_i in dfPLATYPOS.age.unique():
            for i in dfPLATYPOS.orb_sep[dfPLATYPOS.age == age_i].index:
                a = dfPLATYPOS.orb_sep[dfPLATYPOS.age == age_i].loc[i]
                for m in dfPLATYPOS.columns[2:].values:
                    R = dfPLATYPOS.loc[i][m]
                    a_R_age_points.append([a, R, age_i])
                    M_age_values.append(float(m))

        a_R_age_points = np.array(a_R_age_points)
        M_age_values = np.array(M_age_values)
        ####################################################################
        self.radius_jup = self.radius/(const.R_jup/const.R_earth).value
        point_i = (self.distance, self.radius_jup, self.age)
        mass_i_jup = griddata(a_R_age_points, M_age_values, point_i, method='linear')
        self.mass = mass_i_jup * (const.M_jup/const.M_earth).value
        self.mass_jup = mass_i_jup
        
        
    def set_name(self, t_final, initial_step_size, epsilon, K_on, beta_on, evo_track_dict):
        """ function to set the right planet name based on the track specified. This can then be used to check if 
        a particular planet on a particular track has already evolved. """
        self.planet_id = 'planet_a'+str(np.round(self.distance, 3)).replace('.', 'dot') + \
                         '_Mj'+str(np.round(self.mass_jup,3)).replace(".", "dot") + \
                         "_Mcore"+str(int(self.core_mass)) + \
                         '_Mstar' + str(np.round(self.mass_star,3)).replace(".", "dot") + "_K_" + K_on + \
                         "_beta_" + beta_on + \
                         "_track_" + str(evo_track_dict["t_start"]) + "_" + str(evo_track_dict["t_sat"]) + \
                         "_" + str(t_final) + "_" + str(evo_track_dict["Lx_max"]) + \
                         "_" + str(evo_track_dict["dt_drop"]) + "_" + str(evo_track_dict["Lx_drop_factor"])
        
    
    def evolve_forward(self, t_final, initial_step_size, epsilon, K_on, beta_on, evo_track_dict, path_for_saving, planet_folder_id):
        """ Call this function to make the planet evolve. """
        
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
                planet_params = "a,mass,core_mass,radius,age\n"+ str(self.distance) + "," + str(self.mass) + "," \
                                                               + str(self.core_mass) + "," + str(self.radius) + "," + str(self.age)
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
            t, M, R, Lx, evolved_off_grid = mass_planet_RK4_forward_MESA(epsilon=epsilon, K_on="yes", beta_on="yes", planet_object=self, 
                                                       initial_step_size=initial_step_size, t_final=t_final, 
                                                       track_dict=evo_track_dict)
            df = pd.DataFrame({"Time": t, "Mass": M, "Radius": R, "Lx": Lx})
            df.to_csv(path_for_saving+self.planet_id+".txt", index=None)
            print("Saved!")
            self.has_evolved = True
            # evolved_off_grid is for tracking the planets which have moved outside of the mass grid
            if evolved_off_grid == True:
                # create file which indicates that the planet has moved outside of the mass grid
                e = open(path_for_saving+"evolved_off_"+self.planet_id+".txt", "a") 
                evolved_off_info = "t_final, M_final\n" + str(t[-1]) + "," + str(M[-1])
                e.write(evolved_off_info)
                e.close()
                      
    
    def read_results(self, file_path):
        if self.has_evolved == True: # planet exists, has been evolved & results (which are stored in file_path) can be read in
            df = pd.read_csv(file_path+self.planet_id+".txt")
            t, M, R, Lx = df["Time"].values, df["Mass"].values, df["Radius"].values, df["Lx"].values
            return t, M, R, Lx
        else:
            print("Planet has not been evolved & no result file exists.")
    
    
            
    def evolve_backwards(self, t_final, initial_step_size, epsilon, K_on, beta_on, evo_track_dict, path_for_saving):
        contine