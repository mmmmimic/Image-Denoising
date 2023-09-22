#!/usr/bin/env python
# coding: utf-8

import tensorflow

import sys
before = [str(m) for m in sys.modules]


# ------- Variable specific to N2V -------
from n2v.models import N2VConfig, N2V
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from csbdeep.io import save_tiff_imagej_compatible

# ------- Common variable to all ZeroCostDL4Mic notebooks -------
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


# Colors for the warning messages
class bcolors:
  WARNING = '\033[31m'
W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red

#Disable some of the tensorflow warnings
import warnings
warnings.filterwarnings("ignore")
os.system('nvidia-smi')

# Create DataGenerator-object.
datagen = N2V_DataGenerator()

Training_source = "cell/train" #@param {type:"string"}

imgs = datagen.load_imgs_from_directory(directory = Training_source, dims='ZYX')

#@markdown ### Model name and path:
model_name = "test_exp" #@param {type:"string"}  # TODO: XYZ  --> ZYX
model_path = "test_exp" #@param {type:"string"}
number_of_epochs = 1000#@param {type:"number"}

#@markdown Patch size (pixels) and number
patch_size = 64#@param {type:"number"}

patch_height = 64#@param {type:"number"}


#@markdown ###Advanced Parameters

Use_Default_Advanced_Parameters = True #@param {type:"boolean"}

batch_size = 16
number_of_steps = 100
percentage_validation = 10
initial_learning_rate = 0.0004



if os.path.exists(model_path+'/'+model_name):
  print(bcolors.WARNING +"!! WARNING: "+model_name+" already exists and will be deleted in the following cell !!")
  print(bcolors.WARNING +"To continue training "+model_name+", choose a new model_name here, and load "+model_name+" in section 3.3"+W)


#Load one randomly chosen training target file

random_choice=random.choice(os.listdir(Training_source))
x = imread(Training_source+"/"+random_choice)

# Here we check that the input images are stacks
if len(x.shape) == 3:
  print("Image dimensions (z,y,x)",x.shape)

if not len(x.shape) == 3:
  print(bcolors.WARNING + "Your images appear to have the wrong dimensions. Image dimension",x.shape)

#Find image Z dimension and select the mid-plane
Image_Z = x.shape[0]
mid_plane = int(Image_Z / 2)+1

#Find image XY dimension
Image_Y = x.shape[1]
Image_X = x.shape[2]

#Hyperparameters fails
# Here we check that patch_size is divisible by 8
if not patch_size % 8 == 0:
    patch_size = ((int(patch_size / 8)-1) * 8)
    print (bcolors.WARNING + " Your chosen patch_size is not divisible by 8; therefore the patch_size chosen is now:",patch_size)

# Here we check that patch_height is smaller than the z dimension of the image
if patch_height > Image_Z :
  patch_height = Image_Z
  print (bcolors.WARNING + " Your chosen patch_height is bigger than the z dimension of your image; therefore the patch_size chosen is now:",patch_height)

# Here we check that patch_height is divisible by 4
if not patch_height % 4 == 0:
    patch_height = ((int(patch_height / 4)-1) * 4)
    if patch_height == 0:
      patch_height = 4
    print (bcolors.WARNING + " Your chosen patch_height is not divisible by 4; therefore the patch_size chosen is now:",patch_height)


Use_Data_augmentation = True

print("Parameters initiated.")


#Here we display a single z plane

norm = simple_norm(x[mid_plane], percent = 99)

f=plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.imshow(x[mid_plane], interpolation='nearest', norm=norm, cmap='magma')
plt.title('Training source')
plt.axis('off');
plt.savefig('TrainingDataExample_N2V3D.png',bbox_inches='tight',pad_inches=0)


Use_Data_augmentation = True #@param {type:"boolean"}

if Use_Data_augmentation:
  print("Data augmentation enabled")

if not Use_Data_augmentation:
  print("Data augmentation disabled")



if os.path.exists(model_path+'/'+model_name):
  print(bcolors.WARNING +"!! WARNING: Model folder already exists and has been removed !!" + W)
  shutil.rmtree(model_path+'/'+model_name)


#Disable some of the warnings
import warnings
warnings.filterwarnings("ignore")

# Create batches from the training data.
patches = datagen.generate_patches_from_list(imgs, shape=(patch_height, patch_size, patch_size), augment=Use_Data_augmentation)

# Patches are divited into training and validation patch set. This inhibits over-lapping of patches.
number_train_images =int(len(patches)*(percentage_validation/100))
X = patches[number_train_images:]
X_val = patches[:number_train_images]

print(len(patches),"patches created.")
print(number_train_images,"patch images for validation (",percentage_validation,"%).")
print((len(patches)-number_train_images),"patch images for training.")

#Here we automatically define number_of_step in function of training data and batch size
if (Use_Default_Advanced_Parameters):
  number_of_steps= int(X.shape[0]/batch_size) + 1

# creates Congfig object.
config = N2VConfig(X, unet_kern_size=3,
                   train_steps_per_epoch=number_of_steps,train_epochs=number_of_epochs, train_loss='mse', batch_norm=True,
                   train_batch_size=batch_size, n2v_perc_pix=0.198, n2v_patch_shape=(patch_height, patch_size, patch_size),
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, train_learning_rate = initial_learning_rate)

vars(config)

# Create the default model.
model = N2V(config=config, name=model_name, basedir=model_path)


print("Parameters transferred into the model.")
print(config)

# Shows a training batch and a validation batch.
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.imshow(X[0,1,...,0],cmap='magma')
plt.axis('off')
plt.title('Training Patch');
plt.subplot(1,2,2)
plt.imshow(X_val[0,1,...,0],cmap='magma')
plt.axis('off')
plt.title('Validation Patch');



start = time.time()

# the training starts.
history = model.train(X, X_val)

# convert the history.history dict to a pandas DataFrame:
lossData = pd.DataFrame(history.history)

if os.path.exists(model_path+"/"+model_name+"/Quality Control"):
  shutil.rmtree(model_path+"/"+model_name+"/Quality Control")

os.makedirs(model_path+"/"+model_name+"/Quality Control")

# The training evaluation.csv is saved (overwrites the Files if needed).
lossDataCSVpath = model_path+'/'+model_name+'/Quality Control/training_evaluation.csv'
with open(lossDataCSVpath, 'w') as f:
  writer = csv.writer(f)
  writer.writerow(['loss','val_loss', 'learning rate'])
  for i in range(len(history.history['loss'])):
    writer.writerow([history.history['loss'][i], history.history['val_loss'][i], history.history['lr'][i]])


# Displaying the time elapsed for training
dt = time.time() - start
mins, sec = divmod(dt, 60)
hour, mins = divmod(mins, 60)
print("Time elapsed:",hour, "hour(s)",mins,"min(s)",round(sec),"sec(s)")

model.export_TF(name='Noise2Void',
                description='Noise2Void 3D trained using ZeroCostDL4Mic.',
                authors=["You"],
               test_img=X_val[0,...,0], axes='ZYX',
               patch_shape=(patch_size, patch_size))