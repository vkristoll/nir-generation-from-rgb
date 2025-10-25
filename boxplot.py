#Calculation of boxplot values

import os
import pandas as pd
import numpy as np
import time


dirimTonpath="path to metric spectra"
dirimTonurban=sorted(os.listdir(f"{dirimTonpath}/urban"))

dirsave="save output"


name="ssim"

A=np.zeros((len(dirimTonurban),5))

start=time.time()
###urban
for k in range (len(dirimTonurban)):
        im=np.float16(pd.read_csv(f"{dirimTonpath}/urban/{dirimTonurban[k]}").iloc[:].values)
        
        A[k,0]=np.amin(im)
        A[k,1]=np.quantile(im, .25)
        A[k,2]=np.quantile(im, .50)
        A[k,3]=np.quantile(im, .75)        
        A[k,4]=np.amax(im)

np.savetxt( f"{dirsave}/{name}_urban.csv", A, delimiter=',', header=" #min, #q1,#q2,#q3,#max",fmt='%10.8f')       
print("finished urban")    
print(time.time()-start)     
       



