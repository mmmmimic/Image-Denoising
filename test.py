
#!/usr/bin/env python
# coding: utf-8

import tensorflow

import sys
before = [str(m) for m in sys.modules]


from n2v.models import N2VConfig, N2V
from csbdeep.io import save_tiff_imagej_compatible

import numpy as np
from matplotlib import pyplot as plt
import urllib
import os, random
import shutil
import zipfile
from tifffile import imread, imsave
import time
from pathlib import Path
import pandas as pd
import csv
from astropy.visualization import simple_norm


Prediction_model_name = "test_exp"
Prediction_model_path = "test_exp"

Data_folder = "./cell/train" #@param {type:"string"}
Result_folder = "./test_exp/test_exp/logs" #@param {type:"string"}
Use_the_current_trained_model = True #@param {type:"boolean"}
Prediction_model_folder = "test_exp/test_exp" #@param {type:"string"}

#Here we find the loaded model name and parent path
Prediction_model_name = os.path.basename(Prediction_model_folder)
Prediction_model_path = os.path.dirname(Prediction_model_folder)


full_Prediction_model_path = Prediction_model_path+'/'+Prediction_model_name+'/'

Automatic_number_of_tiles = False #@param {type:"boolean"}

#Activate the pretrained model.
config = None
model = N2V(config, Prediction_model_name, basedir=Prediction_model_path)

thisdir = Path(Data_folder)
outputdir = Path(Result_folder)
suffix = '.tif'

# r=root, d=directories, f = files
for r, d, f in os.walk(thisdir):
    for file in f:
        if ".tif" in file:
            print(os.path.join(r, file))

# The code by Lucas von Chamier
n_tiles_Z = 1
n_tiles_Y = 1
n_tiles_X = 1
n_tilesZYX = (n_tiles_Z, n_tiles_Y, n_tiles_X)
for r, d, f in os.walk(thisdir):
  for file in f:
    base_filename = os.path.basename(file)
    input_train = imread(os.path.join(r, file))
    pred_train = model.predict(input_train, axes='ZYX', n_tiles=n_tilesZYX)
    save_tiff_imagej_compatible(os.path.join(outputdir, base_filename), pred_train, axes='ZYX')


#Display an example
random_choice=random.choice(os.listdir(Data_folder))
x = imread(Data_folder+"/"+random_choice)

#Find image Z dimension and select the mid-plane
Image_Z = x.shape[0]
mid_plane = int(Image_Z / 2)+1

y = imread(Result_folder+"/"+random_choice)

f=plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.imshow(x[mid_plane], interpolation='nearest')
plt.title('Noisy Input (single Z plane)');
plt.axis('off');
plt.subplot(1,2,2)
plt.imshow(y[mid_plane], interpolation='nearest')
plt.title('Prediction (single Z plane)');
plt.axis('off');