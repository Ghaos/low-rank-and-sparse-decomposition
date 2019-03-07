# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import os

class ImageLoader:
    #画像の読み込み
    def load(self, root, frames):
        
        image_path = root + 'data/image0001.jpg'
        img = Image.open(image_path)
        img_array = np.asarray(img)
        w, h = img.size                                  #サイズ取得
        D = np.reshape(img_array, (1, w*h))
        
        for i in range(1,frames):
            image_path = root + 'data/image' + format(i+1, '04d') + '.jpg'
            img = Image.open(image_path)
            img_array = np.asarray(img)
            D = np.vstack([D, np.reshape(img_array, (1, w*h))])
            
        return w, h, D
    
    #画像の保存
    def save(self, root, filename, Data, width, height, frames): 
        os.makedirs(root)
        
        for i in range(frames):
            save_path = root + '/' + filename + format(i+1, '04d') + '.png'
            im_array = np.reshape(Data[i,:], (height, width))
            Image.fromarray(im_array.astype(np.uint8)).save(save_path)
            
        return