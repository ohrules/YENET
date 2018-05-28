from cv2 import imread
import cv2
import os
from enet import get_unet
import keras
import numpy as np

def save(data, i, path):
    data = data.astype('float32')
    cv2.imwrite(path + str(i) + ".jpg",data)

def getImage(i, source, main_dir, ext, size):
    name =str(i) + ext #for table tennis data

    path = os.path.join(main_dir, source , name)
    print(path)
    img = imread(path, 0)
    img = cv2.resize(img, size)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype('float32')
    img /= 255
    return img

def getImages(start,videoType):
    i = start
    dir = '..\\testing\\without_feedback\\' + str(videoType) + '\\'
    frame_ext = '.jpg'
    x1 = getImage(i  , '', dir, frame_ext, (224, 224))
    x2 = getImage(i+1, '', dir, frame_ext, (224, 224))
    x3 = getImage(i+2, '', dir, frame_ext, (224, 224))
    print("-------------------------------")

    x1 = x1.reshape((1, x1.shape[0], x1.shape[1], x1.shape[2]))
    x2 = x2.reshape((1, x2.shape[0], x2.shape[1], x2.shape[2]))
    x3 = x3.reshape((1, x3.shape[0], x3.shape[1], x3.shape[2]))

    x1_2 = np.concatenate([x1, x2], axis=3)

    return x1_2, x3, x1, x2

def test():
    md = get_unet()

    md.load_weights('..\\checkpoints\\enet.hdf5')
    root = '..\\testing\\without_feedback\\enet'
    for videoType in range(1,2):
        tc = 1
        for start in range(1,30):

            tag = root + "\\"
            
            
            h, hy, x1, x2 = getImages(start,videoType)
            
            hp = md.predict(h)
            
            if not os.path.exists(tag):
                os.makedirs(tag)
            
            i = start
            for p, g in zip(hp, hy):
                p *= 255;
                g *= 255;
                x1 *= 255;
                x2 *= 255;
                save(p.reshape((224, 224)), i, tag) #output
                #save(g.reshape((224, 224)), i, tag+'x4') #ground truth
                #save(x1.reshape((224, 224)), i, tag+'x1')#input1
                #save(x2.reshape((224, 224)), i, tag+'x2')#input2
                i += 1
	    	
            tc+=1
test()
