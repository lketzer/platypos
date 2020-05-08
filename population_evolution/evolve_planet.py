import os
import multiprocessing as mp
import multiprocessing
import time
import sys
sys.path.append('../plaml_package/')
# Planet Class
from Planet_class_LoFo14 import planet_LoFo14
from Planet_class_Ot20 import planet_Ot20
from Planet_class_LoFo14_PAPER import planet_LoFo14_PAPER
from Planet_class_Ot20_PAPER import planet_Ot20_PAPER


def evolve_one_planet(pl_folder_pair, t_final, init_step, eps, K_on, beta_on, evo_track_dict_list, path_for_saving):
    """Evolves one planet, pl, at a time, but through all evo tracks in evo_track_dict_list. 
    So all calculations for one planet belong to each other, but the planets are independent of each other. 
    Each time this function is called represents an independent process. """
    for track in evo_track_dict_list:
        pl = pl_folder_pair[1]
        folder = pl_folder_pair[0]
        pl.set_name(t_final, init_step, eps, K_on, beta_on, evo_track_dict=track) # set planet name based on specified track
        # generate a useful planet name for saving the results
        pl_file_name = folder + "_track" + pl.planet_id.split("_track")[1] + ".txt"
        if os.path.isdir(path_for_saving+pl_file_name):
            # skip & don't evolve planet
            # print("Folder "+pl_file_name+" exists.")
            pass
        else:
            pl.evolve_forward(t_final, init_step, eps, K_on, beta_on, evo_track_dict=track, 
                              path_for_saving=path_for_saving, planet_folder_id=folder)


def evolve_ensamble(planets_chunks, t_final, initial_step_size, epsilon, K_on, beta_on, evo_track_dict_list, path_save):
    """Function which parallelizes the multiprocessing_func.
    @param planets_chunks: (array of lists) Array of lists of [folder, planet] pairs. 
            Each list (i.e. list of planets) in the array will be run seperately.
    @param t_final: (float)    
    @param initial_step_size: (float)     
    @param epsilon: (float)
    @param K_on: (str)
    @param beta_on: (str)
    @param evo_track_dict_list: (list) List of track_dictionaries along which to evolve each planet
    @param path_save: (str) Filepath to the folder in which to save the planet-folders
    @result: Each planet will evolve along all the specified tracks. 
             Each track-planet evolution result will be saved as a csv file in the planet-folder.
    """
    print("start")
    #if __name__ == '__main__':
    for i in range(len(planets_chunks)):
        starttime = time.time()
        processes = []
        for j in range(len(planets_chunks[i])):
            pl_folder_pair = planets_chunks[i][j] # get folder-planet pair
            path_for_saving = path_save + planets_chunks[i][j][0] + "/" # get corresponding folder
            p = multiprocessing.Process(
                target=evolve_one_planet, args=(pl_folder_pair, t_final, initial_step_size, epsilon, K_on, beta_on, 
                                                evo_track_dict_list,  path_for_saving))
            processes.append(p)
            p.start()

        for process in processes:
            process.join()

        t = (time.time() - starttime)/60
        print('That took {} minutes'.format(t))
