import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import cv2
import os
import glob
from lxml import etree

img_dir="D:/Workspace/pythonProject1/numberplate/numpy-cnn-master/data/car-plate-detection/images/"
def get_data(path):
    IMAGE_SIZE=200

    data_path=os.path.join(img_dir)
    files=os.listdir(data_path)
    print(files)
    print(data_path)
    X=[]
    for i in files:
        img_path=img_dir+i
        img=cv2.imread(img_path)
        # img=cv2.imread(f1)
        img=cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
        X.append(np.array(img))
    X=np.array(X)
    return X

def resizeannotation(f):
    IMAGE_SIZE=200
    tree = etree.parse(f)
    for dim in tree.xpath("size"):
        width = int(dim.xpath("width")[0].text)
        height = int(dim.xpath("height")[0].text)
    for dim in tree.xpath("object/bndbox"):
        xmin = int(dim.xpath("xmin")[0].text)/(width/IMAGE_SIZE)
        ymin = int(dim.xpath("ymin")[0].text)/(height/IMAGE_SIZE)
        xmax = int(dim.xpath("xmax")[0].text)/(width/IMAGE_SIZE)
        ymax = int(dim.xpath("ymax")[0].text)/(height/IMAGE_SIZE)
    return [int(xmax), int(ymax), int(xmin), int(ymin)]

path = 'D:/Workspace/pythonProject1/numberplate/numpy-cnn-master/data/car-plate-detection/annotations'
text_path='D:/Workspace/pythonProject1/numberplate/numpy-cnn-master/data/car-plate-detection/annotations/'
def get_y(path,text_path):

    text_files = [text_path+f for f in sorted(os.listdir(path))]
    y=[]
    for i in text_files:
        y.append(resizeannotation(i))
    y=np.array(y)    
    return y
X=get_data(img_dir)
y=get_y(path,text_path)
print(X.shape)
print(y.shape)
