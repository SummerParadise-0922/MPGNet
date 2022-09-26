from genericpath import exists
from PIL import Image
import numpy as np
from glob import glob
import os

def argument(clear_img:np.array,degraded_img:np.array):
    # flip up and down
    if np.random.randint(0,2) == 1:
        clear_img = clear_img[::-1]
        degraded_img = degraded_img[::-1]
    # Flip left and right
    if np.random.randint(0,2) == 1:
        clear_img = clear_img[:,::-1]
        degraded_img = degraded_img[:,::-1]
    # Rotate 90 degrees counterclockwise
    if np.random.randint(0,2) == 1:
        clear_img = np.rot90(clear_img,1)
        degraded_img = np.rot90(degraded_img,1)
    # Rotate 90 degrees clockwise
    if np.random.randint(0,2) == 1:
        clear_img = np.rot90(clear_img,-1)
        degraded_img = np.rot90(degraded_img,-1)
    return [clear_img,degraded_img]


class pair_dataset(object):
    def __init__(self,fn_root,subdir='Train',category='Poled',is_Train=True):
        self.clear_fns = np.sort(glob(os.path.join(fn_root,subdir,category,'*.png')))
        self.isTrain=is_Train

    def __len__(self):
        return len(self.clear_fns)
    
    def __getitem__(self,index):
        # data read
        clear_fn = self.clear_fns[index]
        degraded_fn = self.clear_fns[index].replace('HQ_patch','LQ_patch')
        clear_img = np.array(Image.open(clear_fn))/255.0
        degraded_img = np.array(Image.open(degraded_fn))/255.0
        # data argument
        if self.isTrain:
            clear_img,degraded_img = argument(clear_img,degraded_img)
        clear_img = np.transpose(clear_img,(2,0,1)).astype(np.float32).copy()
        degraded_img = np.transpose(degraded_img,(2,0,1)).astype(np.float32).copy()
        return {'clear':clear_img,'degraded':degraded_img}

