import os
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import scipy.optimize as optimize
import astropy.units as u
from astropy import constants as const

from platypos.lx_evo_and_flux import flux_at_planet_earth, flux_at_planet
from platypos.mass_evolution_function import mass_evo_RK4_forward


class Planet_ChRo16():
    """
    Need star and planet dictionary to initialize a ChRo planet object.

    Structure of star_dictionary: {'star_id': "dummySun", 'mass': mass_star,
                                   'radius': radius_star, 'age': age_star,
                                   'L_bol': L_bol, 'Lx_age': Lx_age}
    Structure of planet_dict: {"core_mass": m_c, "fenv": f,
                              "distance": a, "core_comp": rock_or_ice}

    NOTE: There are actually three options for planet_dict:
         Case 1) An artificial planet with given core mass and envelope mass
                 fraction (M_core & f_env)
                -> in this case we need to calculate the mass and radius of
                the planet (M_pl & R_pl)

         Case 2) An observed planet with a known mass (we have M_pl & R_pl
                 & f_env) -> M_core

         Case 3) An observed planet with radius and mass measurement, plus
                 core mass is specified
                 -> need to calculate/estimate envelope mass fraction

    Additional Input: Lx_sat_info (only needed if you want to scale the
                      1 & 5 Gyr X-ray luminosities for non-solar-mass stars
    """

    def __init__(self, star_dictionary, planet_dict,
                 Lx_sat_info=None, Lx_sat_Lbol=None):
        '''Initialize a new instance (=object) of Planet_ChRo16

        Parameters:
        -----------


        '''

        # initialize stellar parameters
        try:
            self.star_id = star_dictionary['star_id']
        except:
            self.star_id = "noname"
        try:
            self.radius_star = star_dictionary['radius']
        except:
            self.radius_star = "None"
        
        self.mass_star = star_dictionary['mass']
        self.age = star_dictionary['age']
        self.Lbol = star_dictionary['L_bol'] * const.L_sun.cgs.value
        self.Lbol_solar = star_dictionary['L_bol']
        self.Lx_age = star_dictionary['Lx_age']
        try:
            self.Lx_sat_info = star_dictionary['Lx_sat_info']
            self.Lx_sat_Lbol = star_dictionary['Lx_sat_Lbol']
        except:
            self.Lx_sat_info = Lx_sat_info
            self.Lx_sat_Lbol = Lx_sat_Lbol

        # initialize planet parameters based on the input dictionary
        # the following 3 parameters are the same for all planets!
        self.distance = planet_dict["distance"]
        try:
            self.core_comp = planet_dict["core_comp"]
        except:
            self.core_comp = "rock"
        self.flux = flux_at_planet_earth(self.Lbol, self.distance)
        self.flux_cgs = flux_at_planet(self.Lbol, self.distance)
        self.calculate_eq_temp()

        # flag which tracks if planet has been evolved and result file exists
        self.has_evolved = False
        self.planet_id = "noname" # set planet name later with set_name
        # variable can be set later
        self.EUV_relation = None
        # I add this extra flag to test a planet object for its class type
        self.planet_type = "ChRo16"  
        

        # the remaining parameters depend on the input dictionary (Case 1)
        while True:
            try:
                # check if envelope mass fraction is specified, then Case 1
                planet_dict["fenv"]
                self.planet_info = "Case 1 - artificial planet"\
                                    + " (with M_core & f_env)"
                # Case 1: artificial planet with fenv & M_core given, need to
                #         calculate the total mass and radius
                self.fenv = planet_dict["fenv"]
                self.core_mass = planet_dict["core_mass"]
                self.calculate_core_radius()
                self.calculate_planet_mass()
                self.calculate_planet_radius()
                self.calculate_density_cgs()
                break
            except:
                raise KeyError("Need to specify fenv and M_core!")

        self.calculate_density()
        self.calculate_grav_potential()
        self.calculate_lambda()
        self.calculate_Roche_lobe()
        self.calculate_period()

    # Class Methods
    def calculate_eq_temp(self):
        """ Calculate planetary equilibrium temperature assuming
        a Bond albedo of 0 (all incoming radiation is absorbed). """
        self.t_eq = (((self.Lbol / const.L_sun.cgs.value * const.L_sun.cgs) \
                         / (16 * np.pi * const.sigma_sb.cgs \
                            * (self.distance * const.au.cgs)**2))**(0.25)).value
        
    def calculate_density(self):
        """ Calculate density in cgs. """
        rho = (self.mass * const.M_earth.cgs)\
              / (4./3 * np.pi * (self.radius * const.R_earth.cgs)**3)
        self.density = rho.value
        
    def calculate_period(self):
        """ Approximate period from Kepler's 3rd law. 
            Assuming a circular orbit.
        """ 
        P_sec = (4 * np.pi**2 \
                 / (const.G.cgs * (self.mass_star * const.M_sun.cgs +
                                   self.mass * const.M_earth.cgs))\
                 * (self.distance * const.au.cgs)**3)**(0.5)
        self.period = P_sec.value / (86400)  # in days

    def calculate_lambda(self):
        """ Calculate (restricted) Jeans escape parameter lambda. 
        See e.g. Fossati et al. 2017, or Kubyshkina et al. 2018 
        for details on this parameter.
        """
        M_H = (const.m_p + const.m_e).cgs
        self.lambda_escape = ((const.G.cgs * (self.mass * const.M_earth.cgs) * M_H) \
                              / (const.k_B.cgs * self.t_eq\
                                 * (self.radius * const. R_earth.cgs))).value

    def calculate_Roche_lobe(self):
        """ Calculate planetary Roche lobe radius in units of Earth radii. """
        R_RL_cgs = (self.distance * const.au.cgs)\
                    * ((self.mass * const.M_earth.cgs)\
                       / (3 * ((self.mass * const.M_earth.cgs)\
                             + (self.mass_star * const.M_sun.cgs))))**(1./3)
        self.radius_RL = (R_RL_cgs / const.R_earth.cgs).value
        
    def calculate_grav_potential(self):
        """ Calculate grav. potential in cgs. """
        grav_pot = - const.G.cgs * (self.mass * const.M_earth.cgs)\
                    / (self.radius * const.R_earth.cgs)
        self.grav_potential = grav_pot.value
    
    def calculate_density_cgs(self):
        """ Calculate planetary mean density. """
        self.density = ((self.mass * const.M_earth.cgs) \
                       / (4./3 * np.pi * (self.radius * const.R_earth.cgs)**3)).value
    
    def calculate_planet_mass(self):
        """ Planet mass determined by core and atmosphere mass
            (specified in terms of envelope mass fraction [in %]). """
        self.mass = self.core_mass / (1 - (self.fenv/100))

    def calculate_planet_core_mass(self):
        """ Core mass determined by planet mass and envelope mass
            (specified in terms of envelope mass fraction [%]). """
        self.core_mass = self.mass * (1 - (self.fenv/100))

    def calculate_core_radius(self):
        """M-R relation for rock/iron Earth-like or icy core (no envelope).
        See Chen & Rogers (2016) for details. 
        """
        if self.core_comp == "rock":
            self.core_radius = 0.97 * self.core_mass**0.28
        elif self.core_comp == "ice":
            self.core_radius = 1.27 * self.core_mass**0.27

    def solve_for_fenv(self):
        """ For known core and planet radius, core mass, age and flux,
            solve for envelope mass fraction."""
        raise NotImplementedError("Need to specify an envelope mass fraction!")


    def calculate_R_env(self):
        """ Check planet_models_ChRo16.py for details on input and
            output parameters;
        """
        # Coefficients Rocky-Core Planets
        dict_rock = {"c_0": 0.131, "c_1": -0.348, "c_2": 0.631,
                     "c_3": 0.104, "c_4": -0.179, "c_12": 0.028,
                     "c_13": -0.168, "c_14": 0.008, "c_23": -0.045,
                     "c_24": -0.036, "c_34": 0.031, "c_11": 0.209,
                     "c_22": 0.086, "c_33": 0.052, "c_44": -0.009}
        # Coefficients Ice-Rock-Core Planets
        dict_ice = {"c_0": 0.169, "c_1": -0.436, "c_2": 0.572,
                    "c_3": 0.154, "c_4": -0.173, "c_12": 0.014,
                    "c_13": -0.210, "c_14": 0.006, "c_23": -0.048,
                    "c_24": -0.040, "c_34": 0.031, "c_11": 0.246,
                    "c_22": 0.074, "c_33": 0.059, "c_44": -0.006}

        if self.core_comp == "rock":
            params = dict_rock
        elif self.core_comp == "ice":
            params = dict_ice

        x1 = np.log10(self.mass)
        x2 = np.log10((self.fenv/100) / 0.05)
        x3 = np.log10(self.flux)
        x4 = np.log10((self.age/1e3) / 5)  # convert age to Gyr
        var = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}
        #print(var)
        
        def sum_notation1(i0=1, end=4):
            total_sum = 0.
            for i in range(i0, end+1):
                c_i = params["c_"+str(i)]
                x_i = var["x"+str(i)]
                f = lambda x : c_i*x_i
                total_sum += f(i)
            return total_sum

        def sum_notation2(i0=1, end_i=4, j0=1, end_j=4):
            total_sum = 0.
            for i in range(i0, end_i+1):
                for j in range(j0, end_j+1):
                    try: 
                        c_ij = params["c_"+str(i)+str(j)]
                        x_i = var["x"+str(i)]
                        x_j = var["x"+str(j)]
                        #print(i, j, c_ij, x_i, x_j)
                        f = lambda x : c_ij*x_i*x_j
                        total_sum += f(i)
                    except KeyError:
                        # this key is not defined
                        continue
            return total_sum

        # calculate envelope radius using the two sum functions
        log10_Renv = params["c_0"] +\
                            sum_notation1(1, 4) +\
                            sum_notation2(1, 4, 1, 4)
        R_env = 10**log10_Renv # in Earth radii
        self.R_env = R_env
        return R_env


    def calculate_planet_radius(self):
        """ Check planet_models_ChRo16.py for details"""
        self.radius = self.core_radius + self.calculate_R_env()


    def set_name(self, t_final, initial_step_size,
                 epsilon, K_on, beta_settings, evo_track_dict):
        """ OBSOLETE
        Function to set the right planet name based on the track specified.
        This can then be used to check if a particular planet on a particular
        track has already evolved (i.e. if outpufile exists). """
        self.planet_id = "planet_" \
                         + str(np.round(self.distance, 3)).replace('.', 'dot') \
                         + '_Mcore' \
                         + str(np.round(self.core_mass, 3)).replace(".", "dot" )\
                         + '_fenv' \
                         + str(np.round(self.fenv, 3)) + '_' + self.core_comp \
                         + '_Mstar' \
                         + str(np.round(self.mass_star, 3)).replace(".", "dot") \
                         + "_K_" + K_on + "_beta_" + beta_on \
                         + "_track_" + str(evo_track_dict["t_start"]) + "_" \
                         + str(evo_track_dict["t_sat"]) + "_" + str(t_final) \
                         + "_" + str(evo_track_dict["Lx_max"]) + "_" \
                         + str(evo_track_dict["dt_drop"]) + "_" \
                         + str(evo_track_dict["Lx_drop_factor"])


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
        path = os.path.join(path_for_saving, planet_folder_id + ".txt")
        if not os.path.exists(path):
            with open(path, "w") as p:
                planet_params = "a,core_mass,fenv,mass,radius,age,t_eq,core_comp\n" \
                                 + str(self.distance) + "," \
                                 + str(self.core_mass) + "," \
                                 + str(self.fenv) + "," + str(self.mass) \
                                 + "," + str(self.radius) + "," \
                                 + str(self.age) + "," \
                                 + str(self.t_eq)+ "," + self.core_comp
                p.write(planet_params)

        # create a file (track_params_planet_....txt) which contains
        # the track parameters
        path = os.path.join(path_for_saving, "track_params_" + self.planet_id + ".txt")
        if not os.path.exists(path):
            with open(path, "w") as t:
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
        path = os.path.join(path_for_saving, "host_star_properties.txt") 
        if not os.path.exists(path):
            with open(path, "w") as s:
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
        path = os.path.join(path_for_saving, self.planet_id + "_final.txt") 
        if not os.path.exists(path):
            with open(path, "w") as p:
                # get last element (final time, mass, radius)
                t_final = results_df["Time"].iloc[-1]
                R_final = results_df["Radius"].iloc[-1]
                M_final = results_df["Mass"].iloc[-1]
                Lx_final = results_df["Lx"].iloc[-1]
                f_env_final = ((M_final - self.core_mass) / M_final) * 100  # %
                planet_params = "a,core_mass,time,fenv,mass,radius,Lx,core_comp,track\n" \
                                + str(self.distance) + "," \
                                + str(self.core_mass) + "," \
                                + str(t_final) + "," \
                                + str(f_env_final) + "," \
                                + str(M_final) + "," \
                                + str(R_final) + "," \
                                + str(Lx_final) + "," \
                                + self.core_comp + "," \
                                + self.planet_id
                p.write(planet_params)

        
    def evolve_forward(self,
                       t_final,
                       initial_step_size,
                       epsilon, K_on, beta_settings,
                       evo_track_dict,
                       path_for_saving,
                       planet_folder_id,
                       relation_EUV="Linsky",
                       mass_loss_calc="Elim",
                       fenv_sample_cut=False):
        """ Call this function to make the planet evolve and 
        create file with mass and radius evolution.
        See Mass_evolution_function.py for details on the integration."""

        path = os.path.join(path_for_saving, self.planet_id + ".txt")
        if os.path.exists(path):
            # planet already exists
            self.has_evolved = True
            df = pd.read_csv(path)
        else:
            #print("Planet: ", self.planet_id+".txt")
            # call mass_planet_RK4_forward_LO14 to start the integration
            df = mass_evo_RK4_forward(self,
                                     evo_track_dict,
                                     mass_loss_calc,
                                     epsilon=epsilon,
                                     K_on=K_on,
                                     beta_settings=beta_settings,
                                     initial_step_size=initial_step_size,
                                     t_final=t_final,
                                     relation_EUV=relation_EUV,
                                     fenv_sample_cut=fenv_sample_cut)

            df.to_csv(path, index=None)
            self.has_evolved = True  # set evolved-flag to True
            
        return df
    

    def evolve_forward_and_create_full_output(self,
                                              t_final,
                                              initial_step_size,
                                              epsilon,
                                              K_on,
                                              beta_settings,
                                              evo_track_dict,
                                              path_for_saving,
                                              planet_folder_id,
                                              relation_EUV="Linsky",
                                              mass_loss_calc="Elim",
                                              fenv_sample_cut=False):
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
                                         epsilon, K_on, beta_settings,
                                         evo_track_dict,
                                         path_for_saving,
                                         planet_folder_id,
                                         relation_EUV=relation_EUV,
                                         mass_loss_calc=mass_loss_calc,
                                         fenv_sample_cut=fenv_sample_cut)
        
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
