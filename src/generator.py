import numpy as np
import cv2
from cv2 import imread
import os
import random

arr = []

def getImage(main_dir, source, pre, name, ext):
    path = os.path.join(main_dir, source, pre, name + ext)
    img = imread(path, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = cv2.resize(img, (224, 224))
    img = img.reshape(224, 224, 1)
    img = img.astype('float32')
    img /= 255

    return img

def getRandom(min, max):
    return random.randint(min, max)

def generateENETRandom(folder_limit, main_dir, frame_pre, frame_ext):
    while True:
        folder = getRandom(1, folder_limit)
        source = str(folder)
        
        i = getRandom(1, arr[folder] - 3) #get random starting image from a randomly selected folder
        x1 = getImage(main_dir, source, '', str(i), frame_ext)
        x2 = getImage(main_dir, source, '', str(i+1), frame_ext)
        g  = getImage(main_dir, source, '', str(i+2), frame_ext)

        x1 = x1.reshape((1, x1.shape[0], x1.shape[1], x1.shape[2]))
        x2 = x2.reshape((1, x2.shape[0], x2.shape[1], x2.shape[2]))
        g = g.reshape((1, g.shape[0], g.shape[1], g.shape[2]))

        x12 = np.concatenate([x1, x2], axis=3)

        yield(x12, g)

def generateYNETRandom(folder_limit, main_dir, frame_pre, audio_pre, frame_ext, audio_ext):
    while True:
        folder = getRandom(1, folder_limit)
        source = str(folder)
        
        i = getRandom(1, arr[folder] - 3) #get random starting image from a randomly selected folder
        
        x = getImage(main_dir, frame_pre, source, str(i), frame_ext)
        a = getImage(main_dir, audio_pre, source, str(int(i/24)), audio_ext)
        g = getImage(main_dir, frame_pre, source, str(i+1), frame_ext)

        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
        a = a.reshape((1, a.shape[0], a.shape[1], a.shape[2]))
        g = g.reshape((1, g.shape[0], g.shape[1], g.shape[2]))

        yield([x, a], g)

def generateFCNet(folder_limit):
    while True:
        folder = getRandom(1, folder_limit)
        source = str(folder)
        
        i = getRandom(1, arr[folder] - 1)
        frame_ext = '.png'
        
        e = getImage('..\\data\\fcnet\\e\\', '', 'tc (' + str(i) + ')' , 'x3'+str(i), frame_ext)
        y = getImage('..\\data\\fcnet\\y\\', '', 'tc (' + str(i+1) + ')' , 'x3'+str(i+1), frame_ext)
        g = getImage('..\\data\\fcnet\\e\\', '', 'tc (' + str(i) + ')' , 'x4'+str(i), frame_ext)

        e = e.reshape((1, e.shape[0], e.shape[1], e.shape[2]))
        y = y.reshape((1, y.shape[0], y.shape[1], y.shape[2]))
        g = g.reshape((1, g.shape[0], g.shape[1], g.shape[2]))
        
        yield([e, y], g)
    

def countFolderImages(folder_limit, main_dir):
    arr.append(0) #hack to start indexing from 1, since folder numbering starts from 1
    for i in range(1,folder_limit+1):
        dataPath = main_dir + "\\" + str(i) + "\\"
        a, b, files = os.walk(dataPath).__next__() #a and b not required by us
        arr.append(len(files)-3) #-3 is just to be safe from out of bounds errors
    print(arr)
