# calculate SSIM between the ground truth and the predicted NIR image

import tifffile as tiff
import numpy as np
import math
import os
import time

dirsave="path to save SSIM output"

dirrefpath="path to reference images"
dirref=sorted(os.listdir(dirrefpath))

dirpredpath="path to predicted images"
dirpred=sorted(os.listdir(dirpredpath))


start_time = time.time()
k1=0.01
c1=(k1*1)**3

k2=0.03
c2=(k2*1)**3
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
    
    im_ssim=np.float32(np.zeros((rows,cols)))
    
   

    temp_cov=0
    for i in range (rows-1):
        #print (f"i is: {i}")
        #print("--- %s seconds ---" % (time.time() - start_time)) 
        for j in range (cols-1):  
           
            mean_ref=np.mean(impad_ref[i:i+filter_size,j:j+filter_size]) 
            mean_pred=np.mean(impad_pred[i:i+filter_size,j:j+filter_size])
          
            var_ref=np.var(impad_ref[i:i+filter_size,j:j+filter_size])
            var_pred=np.var(impad_pred[i:i+filter_size,j:j+filter_size])
        
            for k in range (filter_size):
                for l in range (filter_size):                   
                    temp_cov=temp_cov+(impad_ref[i+k,j+l]-mean_ref)*(impad_pred[i+k,j+l]-mean_pred)
            
            cov=temp_cov/((filter_size**2)-1)               
            im_ssim[i,j]=((2*mean_ref*mean_pred+c1)*(2*cov+c2))/(((mean_ref**2)+(mean_pred**2)+c1)*(var_ref+var_pred+c2))           
            temp_cov=0
            
    
    print("--- %s seconds ---" % (time.time() - start_time))  
    tiff.imwrite(f"{dirsave}/{newname}ssim7x7_im10BN.tiff", im_ssim)

