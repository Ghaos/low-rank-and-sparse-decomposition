# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import os

class ImageLoader:
    def load(self, root, frames):
        
        image_path = root + '/data/image0001.jpg'
        img = Image.open(image_path)
        w, h = img.size
        D = np.empty((frames, w*h))
        
        for i in range(frames):
            image_path = root + '/data/image' + format(i+1, '04d') + '.jpg'
            img = Image.open(image_path)
            img_array = np.asarray(img) / 255.0
            D[i,:] = np.reshape(img_array, (1, w*h))
            
        return w, h, D
    
    
    def save(self, root, filename, Data, width, height, frames): 
        if not os.path.isdir(root):
            os.makedirs(root)
        
        for i in range(frames):
            save_path = root + '/' + filename + format(i+1, '04d') + '.png'
            im_array = np.reshape(Data[i,:], (height, width))
            Image.fromarray(im_array.astype(np.uint8)).save(save_path)
            
        return