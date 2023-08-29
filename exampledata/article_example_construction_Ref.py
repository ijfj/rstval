import numpy as np
from osgeo import gdal
import os, glob

x_pixels=100
y_pixels=100

s2_data201806=glob.glob(os.path.join(os.getcwd(), "cut_201806*.tif"))[0] 
print(s2_data201806)

ref_link=gdal.Open(s2_data201806)

driver = gdal.GetDriverByName('GTiff')

cl=np.zeros((x_pixels, y_pixels))

for i in range(x_pixels):
    for j in range(y_pixels):
        ii=int(i/25)
        jj=int(j/25)
        nn=ii+jj*4
        cl[i,j]=nn%3+1

b_data = driver.Create("ref.tif",
                       x_pixels, 
                       y_pixels, 
                       1, 
                       gdal.GDT_Int32)

# Raster com 1 banda ndvi
b_data.GetRasterBand(1).WriteArray(cl)

# Sistema de coordenadas de uma das bandas de input
geotrans=ref_link.GetGeoTransform()
proj=ref_link.GetProjection() 
b_data.SetGeoTransform(geotrans) 
b_data.SetProjection(proj)

b_data.FlushCache()
b_data=None

