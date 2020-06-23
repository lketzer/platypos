import os
import pandas as pd

def read_results_file(path, filename):
    """Function to read in the results file for an individual track. """
    
    df = pd.read_csv(path + filename, float_precision='round_trip') 
    # NOTE: set float_precision to avoid pandas doing sth. weird to the last digit
    # Pandas uses a dedicated dec 2 bin converter that compromises accuracy in preference to speed.
    # Passing float_precision='round_trip' to read_csv fixes this.
    # t, M, R, Lx = df["Time"].values, df["Mass"].values, df["Radius"].values, df["Lx"].values
    
    return df


def read_in_PLATYPOS_results(path_to_results, N_tracks):
    """ Call this function to read in ALL the results.
    
    Parameters:
    -----------
    path_to_results (str): path to the master folder which containts all the results for a single run 
                           (i.e. all the subfolders for the individual planets)
    
    N_tracks (int): total number of tracks specified when running PLATYPOS; only planets with complete output are used
                    (this is mainly so that I can read in and look at the results, even though Platypos is still running)
    
    Returns:
    --------
    planet_df_dict (dict): dictionary of results (final parameters), with planet names as keys, and corresponding 
                           results-dataframes as values (results-dataframes have N_tracks*4 columns [t1,M1,R1,Lx1, etc..])
    
    tracks_dict (dict): dictionary with track infos; keys are planet names, values are a list (of length N_tracks) with the track
                    parameters [track_number, full track name, evolved_off flag (True of False) <- last parameter is only 
                    important for MESA planets]
    
    planet_init_dict (dict): dictionary of initial planet parameters, with planet names are keys, and the intial parameters values
                    (intial planet parameters for LoFo14 planets are: 
                    semi-major axis - a, M_init: mass, R_init: radius, M_core: mass_core, age0: age;
                    where a is in AU, mass, radius and core mass in Earth units, age in Myr)
    """

    files = os.listdir(path_to_results)
    print("Total # of planet folders = ", len(files))
    # check for empty folders (where maybe sth went wrong, or where planet has not evolved yet)
    non_empty_folders = []
    for f in files:
        if len(os.listdir(path_to_results + f)) == 0:
            pass
        elif len([file for file in os.listdir(path_to_results + f) if f in file and "track" in file]) == 0:
            # this means no output file has been produced by PLATYPOS for any of the tracks 
            # (the filenames start with f and contain "track)")
            pass
        else:
            non_empty_folders.append(f)
    print("Non-empty folders: ", len(non_empty_folders))

    # Next, read in the results for each planet
    planet_df_dict = {} # dictionary of results, with planet names as keys, and corresponding results-dataframes as values
    tracks_dict = {} # dictionary with planet names as keys, and list of track infos for each planet folder as values
    planet_init_dict = {} # dictionary of initial planet parameters,with planet names are keys, parameters the values
    
    for i, f in enumerate(non_empty_folders):
        # f: folder name
        # get all files in one planet directory
        all_files_in_f = [f for f in os.listdir(path_to_results+f) if not f.startswith('.')] 
        # NOTE: I had an instance where there were hidden files in a folder, 
        # so this is to make sure hidden files (starting with ".") are ignored!

        # make list of only files which contain calculation results 
        # (they start with the planet-folder name & contain "track", and do not contain these other terms 
        # -> those are other files in the directory)
        result_files = [file for file in all_files_in_f if ("track" in file) \
                                                        and ("track_params" not in file) \
                                                        and ("evolved_off" not in file) \
                                                        and ("final" not in file)]
        
        # sort result files first by t_sat, then by dt_drop, then by Lx_drop_factor; this is to have some order in the results 
        # e.g. planet_001_track_10.0_240.0_5000.0_2e+30_0.0_0.0.txt
        #     planet_name    t_start t_sat t_fina  Lx_sat  Lx_drop  Lx_drop_factor
        result_files_sorted = sorted(sorted(sorted(result_files, key=lambda x: float(x.rstrip(".txt").split("_")[-1])),
                                            key=lambda x: float(x.rstrip(".txt").split("_")[-2])),
                                     key=lambda x: float(x.rstrip(".txt").split("_")[-5]))

        # skip planets which do not have all tracks available! (important for when reading in results while Platypos is still running)
        N_tracks_subfolder = len(result_files_sorted) # number of tracks for which results are available
        # only read in results in a single planet folder if all track-results are available
        if N_tracks_subfolder == N_tracks:
            # get file with initial planet params (name: f+".txt")
            df_pl = pd.read_csv(path_to_results + f + "/" + f + ".txt", float_precision='round_trip')
            planet_init_dict[f] = df_pl.values[0] # add to dictionary

            # build dataframe with results from all tracks
            df_all_tracks = pd.DataFrame()
            # read in the results file for each track one by one and build up one single (one-row) dataframe per planet
            for file in result_files_sorted:
                df_i = read_results_file(path_to_results, f + "/" + file)
                df_all_tracks = pd.concat([df_all_tracks, df_i], axis=1)
                # df.reset_index(level=0)
                
            # set the column names of the new dataframe (i goes from 1 to N_tracks+1)
            col_names = []
            for i in range(1, int(len(df_all_tracks.columns)/4)+1):
                col_names.append("t"+str(i))
                col_names.append("M"+str(i))
                col_names.append("R"+str(i))
                col_names.append("Lx"+str(i))
            df_all_tracks.columns = col_names

            # now I have a dataframe for each planet, which contains all the final parameters for each evolutionary track
            planet_df_dict[f] = df_all_tracks # add to dictionary with planet name as key

            # NEXT, read in track names in case I need this info later, for example for knowing the properties of each track
            # track number corresponds to t1,2,3, etc..
            track_dict = {}
            # flag planets which have moved off (only important for MESA planets for now)
            list_planets_evolved_off = ["track" + file.rstrip(".txt").split("track")[1] for file in all_files_in_f if "evolved_off" in file]
            track_info_list = []
            for i, file in enumerate(result_files_sorted):
                if "track" + file.rstrip(".txt").split("track")[1] in list_planets_evolved_off: 
                    track_evooff = True # if track name in list of planets which has evolved off, set flag to True
                else:
                    track_evooff = False
                track_info_list.append((str(i+1), file[len(f+"_"):].rstrip(".txt"), track_evooff)) # contains all the info for each track
            tracks_dict[f] = track_info_list # add track_dictionary for each planet to a master-track dictionary

        else:
            print("Tracks for ", f, " avaiable: ", str(N_tracks_subfolder) + "/" + str(N_tracks))

    # Lastly, convert the planet_init_dict to a dataframe
    planet_init_df = pd.DataFrame.from_dict(planet_init_dict, orient='index', columns=df_pl.columns)
    
    # now I can easily access the planet name, semi-major axies, core mass, initial mass and radius -
    # -> all info is stored in a beautiful dataframe
    print("\nTotal number of planets to analyze: ", len(planet_init_df))
    
    return planet_df_dict, planet_init_df, tracks_dict


def read_in_host_star_parameters(path_to_results):
    """ Function to read in the initial stellar parameters for each planet in the population. 
    
    Parameter:
    ----------
    path_to_results (str): path to the master folder which containts all the results for a single run 
    
    Returns:
    --------
    star_df (dataframe): dataframe with planet id's as indices, and all initial host star parameters as columns
    """
    
    files = os.listdir(path_to_results)
    
    # only use non-empty folders
    non_empty_folders = []
    for f in files:
        if len(os.listdir(path_to_results+f)) == 0:
            pass
        elif len([file for file in os.listdir(path_to_results + f) if f in file and "track" in file])==0:
            # this means no output file has been produced by PLATYPOS for any of the tracks 
            pass
        else:
            non_empty_folders.append(f)
    print("Non-empty folders: ", len(non_empty_folders))

    # add all host star parameters to a dictionary (with planet names as keys, and star parameters as values)
    star_dict = {} 
    for i, f in enumerate(non_empty_folders):
        # f: folder name
        # get all files in one planet directory
        all_files_in_f = [f for f in os.listdir(path_to_results + f) if not f.startswith('.')] 
        df_star = pd.read_csv(path_to_results+f+"/"+"host_star_properties"+".txt", float_precision='round_trip')
        star_dict[f] = df_star.values[0]

    # convert the planet_init_dict to a dataframe
    star_df = pd.DataFrame.from_dict(star_dict, orient='index', columns=df_star.columns)
    
    return star_df



def read_in_PLATYPOS_results_dataframe(path_to_results, N_tracks):
    """
    Calls read_in_PLATYPOS_results & then does some more re-aranging to the data to make it easier to handle.
    
    Parameters:
    -----------
    path_to_results (str): path to the master folder which containts all the results for a single run 
                           (i.e. all the subfolders for the individual planets)
    
    N_tracks (int): total number of tracks specified when running PLATYPOS; only planets with complete output are used
                    (this is mainly so that I can read in and look at the results, even though Platypos is still running)
    
    Returns:
    --------
    planet_all_df (dataframe): dataframe with planet id's as indices and initial and final parameters (for all the tracks) as columns
    
    tracks_dict (dict): dictionary with planet id's as keys, and list of all tracks and their parameters as values
    
    """
    # call read_in_PLATYPOS_results
    planet_df_dict, planet_init_df, tracks_dict = read_in_PLATYPOS_results(path_to_results, N_tracks)

    # create a master dictionary which contains all final (!) parameters
    planet_final_dict = {} 
    for key_pl, df_pl in planet_df_dict.items():
        N_tracks = int(len(df_pl.columns)/4) # number of tracks for which there is a result file available

        # now I need to check for each track what the index of the last non-nan value is
        # (if planet has moved outside of grid, or has reached the stopping condition, PLATYPOS 
        # terminates & returnes the final planetary parameters. Problem: This might not always be at 
        # the same time step for each track! So when I read in all the tracks into one dataframe, the 
        # remaining rows will be filled with NaNs)

        df_final = pd.DataFrame()
        for i in range(1, N_tracks+1):
            final_index = df_pl["M"+str(i)].last_valid_index() # return index for last non-NA/null value
            # get corresponding final time, mass & radius, add to df_final
            df_final.at[0, "t"+str(i)] = df_pl["t"+str(i)].loc[final_index]
            df_final.at[0, "R"+str(i)] = df_pl["R"+str(i)].loc[final_index]
            df_final.at[0, "M"+str(i)] = df_pl["M"+str(i)].loc[final_index]
        df_final.reset_index(drop=True)
        planet_final_dict[key_pl] = df_final.values[0]

    # convert master dictionary to dataframe with planet id's as indices
    planet_final_df = pd.DataFrame.from_dict(planet_final_dict, orient='index', columns=df_final.columns)

    # concatenate planet_init_df and planet_final_df into one master dataframe
    planet_all_df = pd.concat([planet_init_df, planet_final_df], axis=1, sort=False)
    # together with tracks_dict this allows me to select, filter, analyze any track and any planet

    # For MESA planets only: to make life easier, add evolved_off info to my master dataframe
    # add N_tracks more columns to dataframe planet_all_df which indicate whether the planet on given track has evolved off or not
    for key_pl in planet_all_df.index.values:
        track_number = [a_tuple[0] for a_tuple in tracks_dict[key_pl]]
        track_evooff = [a_tuple[2] for a_tuple in tracks_dict[key_pl]]
        for number, evooff in zip(track_number, track_evooff):
            planet_all_df.at[key_pl, "track"+str(number)] = evooff

    return planet_all_df, tracks_dict


def read_in_PLATYPOS_results_dataframe_second_to_last_time(path_to_results, N_tracks):
    """
    The same as read_in_PLATYPOS_results_dataframe, but returns a dataframe with the final parameters one time step before 
    Platypos terminated! This is to analyze the behavior of my code. 
    """
    
    # call read_in_PLATYPOS_results
    planet_df_dict, planet_init_df, tracks_dict = read_in_PLATYPOS_results(path_to_results, N_tracks)

    # create a master dictionary which contains all final (!) parameters
    planet_final_dict = {} 
    for key_pl, df_pl in planet_df_dict.items():
        N_tracks = int(len(df_pl.columns)/4) # number of tracks for which there is a result file available

        # now I need to check for each track what the index of the last non-nan value is
        # (if planet has moved outside of grid, or has reached the stopping condition, PLATYPOS 
        # terminates & returnes the final planetary parameters. Problem: This might not always be at 
        # the same time step for each track! So when I read in all the tracks into one dataframe, the 
        # remaining rows will be filled with NaNs)

        df_final = pd.DataFrame()
        for i in range(1, N_tracks+1):
            final_index = df_pl["M"+str(i)].last_valid_index() # return index for last non-NA/null value
            # get corresponding final time, mass & radius, add to df_final
            df_final.at[0, "t"+str(i)] = df_pl["t"+str(i)].loc[final_index-1]
            df_final.at[0, "R"+str(i)] = df_pl["R"+str(i)].loc[final_index-1]
            df_final.at[0, "M"+str(i)] = df_pl["M"+str(i)].loc[final_index-1]
        df_final.reset_index(drop=True)
        planet_final_dict[key_pl] = df_final.values[0]

    # convert master dictionary to dataframe with planet id's as indices
    planet_final_df = pd.DataFrame.from_dict(planet_final_dict, orient='index', columns=df_final.columns)

    # concatenate planet_init_df and planet_final_df into one master dataframe
    planet_all_df = pd.concat([planet_init_df, planet_final_df], axis=1, sort=False)
    # together with tracks_dict this allows me to select, filter, analyze any track and any planet

    # For MESA planets only: to make life easier, add evolved_off info to my master dataframe
    # add N_tracks more columns to dataframe planet_all_df which indicate whether the planet on given track has evolved off or not
    for key_pl in planet_all_df.index.values:
        track_number = [a_tuple[0] for a_tuple in tracks_dict[key_pl]]
        track_evooff = [a_tuple[2] for a_tuple in tracks_dict[key_pl]]
        for number, evooff in zip(track_number, track_evooff):
            planet_all_df.at[key_pl, "track"+str(number)] = evooff

    return planet_all_df, tracks_dict

