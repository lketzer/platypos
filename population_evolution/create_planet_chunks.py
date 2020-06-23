import os
import math
import numpy as np
import copy

def create_planet_chunks(curr_path, folder_name, list_planets, chunk_size):
    """  
    Function to create the file structure for saving the results, 
    and to split up a (possibly) long list of planet objects into a list of sublists with N elements each (multiprocessing preparation).
    
    NOTE: This was necessary because otherwise the multiprocessing function would make my memory overflow.
    Probably there is a better solution out there but this workaround works.
    
    Parameters:
    -----------
    curr_path (str): file path to the directory where the results of the run should be saved

    folder_name (str): name of the master folder in which to save the individual results in (will be created in curr_path if not existent)
            
    list_planets (list): list of planet objects to evolve
            
    chunk_size (int): Number of planet objects in each sublist (chunk_size=8 seems to work fine)
    
    Returns:
    --------
    path_save (str): curr_path + folder_name
    
    planets_chunks (list): list of [folder-planet]-pair lists
    """
    
    # First, check if directroy for saving the results exists, if not create one
    path_save = curr_path + folder_name
    if os.path.isdir(path_save):
        print("Folder " + folder_name + " exists.")
        pass
    else:
        os.mkdir(path_save)
        
    # Next, create subfolders - one for each planet in the population 
    # (what this means: two planets with the exact same parametes will get two different folders)
    # for consitstent folder names fill string with zeros to have same length
    digits_of_planet_number = len(str(len(list_planets))) 
    for i in range(len(list_planets)):
        planet_number_str = str(i+1) # start planet number at 1
        planet_id = "planet_" + planet_number_str.zfill(digits_of_planet_number) # e.g. 1000 planets -> 'planet_0001' is the first folder name
        # if exists, skip, else create the folder
        if os.path.isdir(path_save+planet_id):
            pass
        else:
            os.mkdir(path_save+planet_id)

   
    # create a dictionary with planet subfolder names as key, and corresponding planet objects as value (one planet belongs into one seperate folder)
    # (e.g. {"planet0001": planet_object_instance})
    result_folders = os.listdir(path_save) # get the individual planet folder names
    planets_dict = {} 
    for i in range(len(list_planets)):
        planets_dict[result_folders[i]] = list_planets[i]
            
    # Lastly, in pareparation for the multiprocessing, split the dictionary into smaller bits and store the key, value pair in a list
    # (e.g. [[["planet1", planet_object1], ["planet2", planet_object2], etc...]])
    # -> I found that if I pass all planets at once to the multiprocessing function, it fills up my memory
    # As a workaround, I only multiprocess a smaller number of planets at a given time (at least this is what I think I'm doing)
    number_of_splits = math.ceil(len(planets_dict)/chunk_size)
    planets_arr = [[key, value] for key, value in planets_dict.items()]
    planets_chunks = np.array_split(planets_arr, number_of_splits)
    
    # now I have an list of [folder-planet] pair lists, which I can individually pass to the multiprocessing_func 
    return path_save, planets_chunks


def create_file_structure_for_saving_results_and_return_planet_folder_pairs(curr_path, folder_name, list_planets):
    """  
    Function to create the file structure for saving the results, 
    and to create a list with folder-planet pairs for the multiprocessing in the next step.
    
    Parameters:
    -----------
    curr_path (str): file path to the directory where the results of the run should be saved

    folder_name (str): name of the master folder in which to save the individual results in (will be created in curr_path if not existent)
            
    list_planets (list): list of planet objects to evolve
    
    Returns:
    --------
    path_save (str): curr_path + folder_name
    
    list_of_folder_planet_pair (list): list of [folder-planet]-pairs   
    """
    
    # First, check if directroy for saving the results exists, if not create one
    path_save = curr_path + folder_name
    if os.path.isdir(path_save):
        print("Folder " + folder_name + " exists.")
        pass
    else:
        os.mkdir(path_save)
        
    # Next, create subfolders - one for each planet in the population 
    # (what this means: two planets with the exact same parametes will get two different folders)
    # for consitstent folder names fill string with zeros to have same length
    digits_of_planet_number = len(str(len(list_planets))) 
    for i in range(len(list_planets)):
        planet_number_str = str(i+1) # start planet number at 1
        planet_id = "planet_" + planet_number_str.zfill(digits_of_planet_number) # e.g. 1000 planets -> 'planet_0001' is the first folder name
        # if exists, skip, else create the folder
        if os.path.isdir(path_save+planet_id):
            pass
        else:
            os.mkdir(path_save+planet_id)

   
    # create a dictionary with planet subfolder names as key, and corresponding planet objects as value (one planet belongs into one seperate folder)
    # (e.g. {"planet0001": planet_object_instance})
    result_folders = os.listdir(path_save) # get the individual planet folder names
    planets_dict = {} 
    for i in range(len(list_planets)):
        planets_dict[result_folders[i]] = list_planets[i]
            
    # create list of [folder, planet object]-pairs
    planets_arr = [[key, value] for key, value in planets_dict.items()]
    
    # now I have an list of [folder-planet] pair lists, which I can individually pass to the multiprocessing_func 
    return path_save, planets_arr


def make_folder_planet_track_list(folder_planet_pair_list, track_list):
    """ takes [folder, planet]-pairs and a list of evolutionary tracks, and combines them into a list of triplets. 
    Output is a list of length len(folder_planet_pair_list)*len(track_list). """
    
    folder_planet_track_list = []
    for i, f_p_pair in enumerate(folder_planet_pair_list):
        #folder_planet_track_list += [[f_p_pair[0], copy.deepcopy(f_p_pair[1]), track_dict] for track_dict in track_list]
        folder_planet_track_list += [[f_p_pair[0], f_p_pair[1], track_dict] for track_dict in track_list]
    
    return folder_planet_track_list