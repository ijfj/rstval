import validation as v
import numpy as np 
import os, glob
import geopandas as gpd

## DATA
#Strata=Classes
cl_name=glob.glob(os.path.join(os.getcwd(), "exampledata/cl*.tif"))[0] 
cl_counts=v.ncountraster(cl_name) 
Ncl = sum(cl_counts.values()) 
#Strata!=Classes 
str_name=glob.glob(os.path.join(os.getcwd(), "exampledata/str1*.tif"))[0] 
str_counts=v.ncountraster(str_name) 
Nstr = sum(str_counts.values()) 
#Shapefile
shp_name=glob.glob(os.path.join(os.getcwd(), "exampledata/val.shp"))[0] 
shp_df = gpd.read_file(shp_name) 

cl_v=np.array(shp_df['cl'].tolist())
ref_v=np.array(shp_df['ref'].tolist())
str_v=np.array(shp_df['str'].tolist())
unitarea=100


#CASE 1
# Stratification without Classes
cmh3d_count, str_kv, report3d = v.metrics(cl_v, ref_v, Ncl, areaunit=100, 
                                          stratified=True, str_counts=str_counts,
                                          strata_classes=False, str_v=str_v) 
str_kv.to_excel('exampleresults/Example1_key-value_Strat.xlsx', header=True)
cmh3d_count.to_excel('exampleresults/Example1_CM.xlsx', header=True)
report3d.to_excel('exampleresults/Example1_Report.xlsx', header=True)

# CASE 2
# 2a
# Stratification with classes. No proportional Sampling. Using Counts
cm2d_count, report2d = v.metrics(cl_v, ref_v, Ncl, areaunit=100, 
                                 stratified=True, str_counts=cl_counts,
                                 strata_classes=True, str_v=None,  
                                 proportional_sampling=False, 
                                 proportions=False)
cm2d_count.to_excel('exampleresults/Example2_CM.xlsx', header=True)
report2d.to_excel('exampleresults/Example2_Report.xlsx', header=True)

# 2b
# Stratification with classes. No proportional Sampling. Using Proportions
cm2d_count, cm2d_prop, report2d = v.metrics(cl_v, ref_v, Ncl, areaunit=100, 
                                            stratified=True, str_counts=cl_counts,
                                            strata_classes=True, str_v=None,  
                                            proportional_sampling=False, 
                                            proportions=True)
cm2d_count.to_excel('exampleresults/Example3_CM.xlsx', header=True)
cm2d_prop.to_excel('exampleresults/Example3_CM_proportions.xlsx', header=True)
report2d.to_excel('exampleresults/Example3_Report.xlsx', header=True)


## DATA 2
#Strata=Classes
cl_name=glob.glob(os.path.join(os.getcwd(), "exampledata/cl*.tif"))[0] 
cl_counts=v.ncountraster(cl_name)
N = sum(cl_counts.values()) 
#Shapefile
shp_name=glob.glob(os.path.join(os.getcwd(), "exampledata/val2.shp"))[0] 
shp_df = gpd.read_file(shp_name)
cl_v=np.array(shp_df['cl'].tolist())
ref_v=np.array(shp_df['ref'].tolist()) 

# CASE 3
# 3a
# Stratification with classes. Proportional Sampling. Using Counts
cm2d_count, report2d = v.metrics(cl_v, ref_v, N, areaunit=100,
                                 stratified=True, str_counts=cl_counts,
                                 strata_classes=True,
                                 proportional_sampling=True,
                                 proportions=False)
cm2d_count.to_excel('exampleresults/Example4_CM.xlsx', header=True)
report2d.to_excel('exampleresults/Example4_Report.xlsx', header=True)

# 3b
# Stratification with classes. Proportional Sampling. Using Proportions
cm2d_count, cm2d_prop, report2d = v.metrics(cl_v, ref_v, N, areaunit=100,
                                            stratified=True, str_counts=cl_counts,
                                            strata_classes=True,
                                            proportional_sampling=True,
                                            proportions=True)
cm2d_count.to_excel('exampleresults/Example5_CM.xlsx', header=True)
cm2d_prop.to_excel('exampleresults/Example5_CM_proportions.xlsx', header=True)
report2d.to_excel('exampleresults/Example5_Report.xlsx', header=True)

# CASE 4
# No stratification
cm2d_count, report2d = v.metrics(cl_v, ref_v, N, areaunit=100)
cm2d_count.to_excel('exampleresults/Example6_CM.xlsx', header=True)
report2d.to_excel('exampleresults/Example6_Report.xlsx', header=True)

