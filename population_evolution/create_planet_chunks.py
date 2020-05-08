import os
import math
import numpy as np

def create_planet_chunks(curr_path, folder_name, list_planets, chunk_size):
    """ Function creates the directory & subdirectory structure for saving the results 
    for all the planets in "list_planets". In addition, it divides list_planets into smaller chunks. 
    This is needed if "list_planets" is very long, and avoids the multiprocessing-action in the evolve_ensamble-function from crashing.
    
    Return:
    ------
    path_save: path specifying where to save the results
    planets_chunks: chunked-up list of planets, which can be passed to "evolve_ensemble" to evolve
    """
    
    path_save = curr_path+folder_name
    if os.path.isdir(path_save):
        # check if directroy for saving the results existst, if not create
        print("Folder -> "+folder_name+" <- exists.")
        pass
    else:
        os.mkdir(path_save)
        
    # create folders, one for each planet in the population 
    # (what this means: two planets with the exact same parametes will get two different folders)
    digits_of_planet_number = len(str(len(list_planets)))
    for i in range(len(list_planets)):
        planet_number_str = str(i+1) # start planet number at 1
        planet_id = "planet_"+planet_number_str.zfill(digits_of_planet_number)
        if os.path.isdir(path_save+planet_id):
            #print("Folder "+planet_id+" exists.")
            pass
        else:
            os.mkdir(path_save+planet_id)

    result_folders = os.listdir(path_save)
    planets_dict = {} # dictionary with folder-planet pairs (one planet belongs into one seperate folder)
    for i in range(len(list_planets)):
        planets_dict[result_folders[i]] = list_planets[i]
            
    # divide dictionary into smaller bits -> I found that if I pass all planets at once for multiprocessing, it fills up my memory
    # this is why I only multiprocess a smaller number at a given time (at least this is what I think I'm doing)
    number_of_splits = math.ceil(len(planets_dict)/9)
    planets_arr = [[key, value] for key, value in planets_dict.items()]
    planets_chunks = np.array_split(planets_arr, number_of_splits)
    # now I have an list of [folder-planet] pairs, which I can pass to the multiprocessing_func
    return path_save, planets_chunks