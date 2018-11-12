# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom 
import png
import itertools
from resizeimage import resizeimage
from PIL import Image
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from glob import glob
from tqdm import tqdm



RSNA_TEST_IMG = "A1_Pneumonia_Detection_DATA/stage_2_test_images/"
RSNA_TRAIN_IMG = "A1_Pneumonia_Detection_DATA/stage_2_train_images/"
SEG_TRAIN_DIRECTORY = "A1_Pneumonia_Detection_PPDATA/stage_2_train_images_png/"
SEG_TEST_DIRECTORY = "A1_Pneumonia_Detection_PPDATA/stage_2_test_images_png/"
SEG_TRAIN_FINAL = "A1_Pneumonia_Detection_PPDATA/SEG_TRAIN/"
SEG_TEST_FINAL = "A1_Pneumonia_Detection_PPDATA/SEG_TEST/"


if not os.path.exists(SEG_TEST_DIRECTORY):
	os.makedirs(SEG_TEST_DIRECTORY)
else: 
	shutil.rmtree(SEG_TEST_DIRECTORY)
	os.makedirs(SEG_TEST_DIRECTORY)

if not os.path.exists(SEG_TRAIN_DIRECTORY):
	os.makedirs(SEG_TRAIN_DIRECTORY)
else: 
	shutil.rmtree(SEG_TRAIN_DIRECTORY)
	os.makedirs(SEG_TRAIN_DIRECTORY)
	
if not os.path.exists(SEG_TEST_FINAL):
	os.makedirs(SEG_TEST_FINAL)
else: 
	shutil.rmtree(SEG_TEST_FINAL)
	os.makedirs(SEG_TEST_FINAL)

if not os.path.exists(SEG_TRAIN_FINAL):
	os.makedirs(SEG_TRAIN_FINAL)
else: 
	shutil.rmtree(SEG_TRAIN_FINAL)
	os.makedirs(SEG_TRAIN_FINAL)
	
def DICOM_to_png(path,destination):
	ds = pydicom.dcmread(path)

	shape = ds.pixel_array.shape

	# Convert to float to avoid overflow or underflow losses.
	image_2d = ds.pixel_array.astype(float)

	# Rescaling grey scale between 0-255
	image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

	# Convert to uint
	image_2d_scaled = np.uint8(image_2d_scaled)

	# Write the PNG file
	with open(destination, 'wb') as png_file:
	    w = png.Writer(shape[1], shape[0], greyscale=True)
	    w.write(png_file, image_2d_scaled)

	#Resize PNG for use with model  
	with open(destination, 'r+b') as f:
	    with Image.open(f) as image:
	        cover = resizeimage.resize_contain(image, [512, 512, 3])
	        cover.save(destination, image.format)
	        
	        
	        
for filename in tqdm(os.listdir(RSNA_TEST_IMG)):
    
    DICOM_to_png((RSNA_TEST_IMG + filename), SEG_TEST_DIRECTORY + os.path.splitext(filename)[0] + ".png")


for filename in tqdm(os.listdir(RSNA_TRAIN_IMG)):
    
    DICOM_to_png((RSNA_TRAIN_IMG + filename), SEG_TRAIN_DIRECTORY + os.path.splitext(filename)[0] + ".png")


def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])
 
 
   
def test_load_image(test_file, target_size):
    img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)
    return img
    
def test_generator(test_files, target_size):
	while True: 	
		for test_file in test_files:
			yield test_load_image(test_file, target_size)    

def save_result(save_path, npyfile, test_files):
    for i, item in enumerate(npyfile):
        result_file = test_files[i]
        img = (item[:, :, 0] * 255.).astype(np.uint8)

        filename, fileext = os.path.splitext(os.path.basename(result_file))

        result_file = os.path.join(save_path, "%s_predict%s" % (filename, fileext))

        cv2.imwrite(result_file, img)
        
test_files = [test_file for test_file in glob(os.path.join(SEG_TEST_DIRECTORY, "*.png")) \
          if ("_mask" not in test_file \
              and "_dilate" not in test_file \
              and "_predict" not in test_file)]

train_files = [test_file for test_file in glob(os.path.join(SEG_TRAIN_DIRECTORY, "*.png")) \
          if ("_mask" not in test_file \
              and "_dilate" not in test_file \
 		          and "_predict" not in test_file)]

model = unet(input_size=(512,512,1))

model.load_weights("unet_lung_seg.hdf5")

model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, \
                  metrics=[dice_coef, 'binary_accuracy'])
                 
                  
                  
test_gen = test_generator(test_files, target_size=(512,512))

train_gen = test_generator(train_files, target_size=(512,512))

test_results = model.predict_generator(test_gen, len(test_files), verbose=1)
train_results = model.predict_generator(train_gen, len(train_files), verbose=1)

save_result(SEG_TEST_FINAL, test_results, test_files)
save_result(SEG_TRAIN_FINAL, train_results, train_files)

# Resave files as segmented overlays 
for filename in tqdm(os.listdir(SEG_TEST_FINAL)):
    predict_image = cv2.imread(SEG_TEST_FINAL + filename)
    mask_image_gray = cv2.cvtColor(predict_image, cv2.COLOR_BGR2GRAY)
    image = cv2.imread(SEG_TEST_DIRECTORY + filename.replace('_predict',''))
    mask = cv2.bitwise_and(image, image, mask=mask_image_gray)
    plt.imsave(SEG_TEST_FINAL + filename ,mask, cmap=plt.cm.bone)
   
for filename in tqdm(os.listdir(SEG_TRAIN_FINAL)):
    predict_image = cv2.imread(SEG_TRAIN_FINAL + filename)
    mask_image_gray = cv2.cvtColor(predict_image, cv2.COLOR_BGR2GRAY)
    image = cv2.imread(SEG_TRAIN_DIRECTORY + filename.replace('_predict',''))
    mask = cv2.bitwise_and(image, image, mask=mask_image_gray)
    plt.imsave(SEG_TRAIN_FINAL + filename ,mask, cmap=plt.cm.bone)




