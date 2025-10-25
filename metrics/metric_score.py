# calculate the metric score
import numpy as np
import tifffile as tiff
from osgeo import gdal
import time

import os

dir1path="path to metric image"
dir1=sorted(os.listdir(dir1path))

dir2path="path to category mask (e.g. urban)"
dir2=sorted(os.listdir(dir2path))

name="SSIM_score"
dirsave="/home/rslab1/Documents/paper_NIR/implementation/ssim_statspred"

start=time.time()
#urban
M2=np.zeros((len(dir1),2))

for k in range (len(dir1)):
    print(f"urban k is {k}")
    #read image
    im=tiff.imread(f"{dir1path}/{dir1[k]}")
    #read mask
    ds=gdal.Open(f"{dir2path}/{dir2[2*k]}") 
    mask=np.array(ds.ReadAsArray())
    
    rows=mask.shape[0]
    cols=mask.shape[1]

    L=[]
    
    for i in range(rows):
        for j in range(cols):
            if mask[i,j]==1:
                L.append(im[i,j])   

    L_ar=np.asarray(L)

    M2[k,0]=np.mean(L_ar)
    M2[k,1]=np.std(L_ar)

np.savetxt( f"{dirsave}/{name}_SSIMscore_urban.csv", M2, delimiter=',', header=" #mean,  #std",fmt='%10.8f')
np.savetxt( f"{dirsave}/{name}_SSIMscore_urbanspectra.csv", L_ar, delimiter=',', header=" #spectra",fmt='%10.8f')
print(time.time()-start)
    
