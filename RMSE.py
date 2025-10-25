# calculate RMSE between the ground truth and the predicted NIR image
import tifffile as tiff
import numpy as np
import math
import os
import time

dirsave="path to save RMSE output"

dirrefpath="path to reference images"
dirref=sorted(os.listdir(dirrefpath))

dirpredpath="path to predicted images"
dirpred=sorted(os.listdir(dirpredpath))

start_time = time.time()
for knum in range (len(dirref)):
    print(f" knum is {knum}")
    #read ref
    imref=tiff.imread(f"{dirrefpath}/{dirref[knum]}")[:,:,3]

    #read pred
    impred=np.squeeze(tiff.imread(f"{dirpredpath}/{dirpred[knum]}"))
    
    name = os.path.basename(f"{dirpred[knum]}")
    newname = name.split('.')[0]  
    
    shape=np.shape(impred)
    rows=shape[0]
    cols=shape[1] 
    
    filter_size=7
    pad=math.floor(filter_size/2)

    impad_ref=np.float32(np.zeros((rows+2*pad,cols+2*pad)))
    impad_ref[pad:rows+pad,pad:cols+pad]=imref

    impad_pred=np.float32(np.zeros((rows+2*pad,cols+2*pad)))
    impad_pred[pad:rows+pad,pad:cols+pad]=impred 
    
    im_rmse=np.float32(np.zeros((rows,cols)))

    temp=0
    for i in range (rows-1):
        #print (f"i is: {i}")
        for j in range (cols-1):
            for k in range (filter_size):
                for l in range (filter_size):                 
                    temp=temp+(impad_ref[i+k,j+l]-impad_pred[i+k,j+l])**2                
            im_rmse[i,j]=math.sqrt(temp/(filter_size**2))      
            temp=0 
            
    
    print("--- %s seconds ---" % (time.time() - start_time))  
    tiff.imwrite(f"{dirsave}/{newname}rmse7x7.tiff", im_rmse)
    
