import os
import pandas as pd

def create_summary_files_with_final_planet_parameters(path_save):
    
    files = os.listdir(path_save)
    non_empty_folders = []
    for f in files:
        if (len(os.listdir(path_save+f)) == 0) and ('.ipynb_checkpoints' in path_save+f):
            pass
        if  len([file for file in os.listdir(path_save+f) if file.split("_")[0]=="planet" and "track" in file]) == 0:
            print("no result files available for {}".format(f))
        else:
            non_empty_folders.append(f)
    print("non-empty folders: ", len(non_empty_folders))


    for i in range(len(non_empty_folders)):
        f = non_empty_folders[i] # folder name: planet_XXXX
        files_in_subfolder = os.listdir(path_save+f)

        # get file with initial planet params
        pl_file = [file for file in files_in_subfolder if file==f+".txt"][0]
        #print(pl_file)
        df_pl = pd.read_csv(path_save+f+"/"+pl_file, float_precision='round_trip')

        # use only files which contain final calculation results 
        # (they start with "planet" & contain "final", but not "params")
        files_in_subfolder = [file for file in files_in_subfolder if ("final" not in file)  and (f in file) 
                           and ("track" in file) and ("track_params" not in file) and ("evolved_off" not in file)]
        #print("\n", np.array(files_in_subfolder), "\n")

        # sort files first by t_sat, then by dt_drop, then by Lx_drop_factor
        files_in_subfolder = sorted(sorted(sorted(files_in_subfolder,
                                                  key=lambda x: float(x.rstrip(".txt").split("_")[-1])),
                                           key=lambda x: float(x.rstrip(".txt").split("_")[-2])),
                                    key=lambda x: float(x.rstrip(".txt").split("_")[-5]))
        #print("\n", np.array(files_in_subfolder), "\n")

        # combine the end results for each track in one summary file
        df = pd.DataFrame(columns=["Time","Mass","Radius","Lx","track"])
        for file in files_in_subfolder:
            #print(file)
            df_i = pd.read_csv(path_save+f+"/"+file, float_precision='round_trip')
            #print(df_i.loc[df_i.index.max()])
            dummy = df_i.loc[df_i.index.max()]
            dummy["track"] = file
            df = df.append(dummy, ignore_index=True)

        #print(path_save+f+"/"+f+"_final.txt")
        if os.path.exists(path_save+f+"/"+f+"_final.txt"):
            try:
                pd.read_csv(path_save+f+"/"+f+"_final.txt")
            except:
                # file exists but is empty -> create!
                #print(df)
                df.to_csv(path_save+f+"/"+f+"_final.txt", index=False)      
        else:
            df.to_csv(path_save+f+"/"+f+"_final.txt", index=False)