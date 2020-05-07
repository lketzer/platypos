# function to write to file

def write_results_to_file(t_arr1, M_arr1, R_arr1, Lx_arr1,
                          t_arr2, M_arr2, R_arr2, Lx_arr2,
                          t_arr3, M_arr3, R_arr3, Lx_arr3,
                          radius_planet, path, filename):
    """col 1-4 is track 1, col 5-8 is track 2 and col 9-12 is track 3"""
    df1 = pd.DataFrame({"Time1": t_arr1, "Mass1": M_arr1, "Radius1": radius_planet(M_arr1), "Lx1": Lx_arr1})
    df2 = pd.DataFrame({"Time2": t_arr2, "Mass2": M_arr2, "Radius2": radius_planet(M_arr2), "Lx2": Lx_arr2})
    df3 = pd.DataFrame({"Time3": t_arr3, "Mass3": M_arr3, "Radius3": radius_planet(M_arr3), "Lx3": Lx_arr3})
    result = pd.concat([df1, df2, df3], axis=1, sort=False)
    result.to_csv(path+filename)
    return None