import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import pydicom
import pandas as pd 
import glob
from sklearn.model_selection import KFold
import pickle 
import tensorflow
import keras.backend

ROOTDIRECTORY = '/usr3/graduate/astoycos/A1_Pneumonia_Model/'

os.chdir('Mask_RCNN')

# Import Mask RCNN

sys.path.append(os.path.join(ROOTDIRECTORY, 'Mask_RCNN'))

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.jpg')
    return list(set(dicom_fps))

class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    
    # Give the configuration a recognizable name  
    NAME = 'pneumonia'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
   
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 4
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.78  ## match target distribution
    DETECTION_NMS_THRESHOLD = 0.01
    LEARNING_MOMENTUM = 0.85
    
    STEPS_PER_EPOCH = 300

config = DetectorConfig()
config.display()

ORIG_SIZE = 1024 

# select trained model 
dir_names = next(os.walk(ROOTDIRECTORY))[1]
key = config.NAME.lower()
dir_names = filter(lambda f: f.startswith(key), dir_names)
dir_names = sorted(dir_names)

if not dir_names:
    import errno
    raise FileNotFoundError(
        errno.ENOENT,
        "Could not find model directory under {}".format(self.model_dir))
    
fps = []

# Pick last directory
for d in dir_names: 
    dir_name = os.path.join(ROOTDIRECTORY, d)
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        print('No weight files in {}'.format(dir_name))
    else: 
      
      checkpoint = os.path.join(dir_name, checkpoints[-1])
      fps.append(checkpoint)

model_path = sorted(fps)[-1]
print('Found model {}'.format(model_path))


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOTDIRECTORY)

# Load trained weights (fill in path to trained weights here)
assert model_path != "" "/Users/andrewstoycos/Documents/Classes_Fall_2018/EC601/MainProject/ec601-project/MASKrcnn_model/mask_rcnn_pneumonia_0099.h5"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors

test_image_fps = get_dicom_fps(ROOTDIRECTORY + 'test_jpegs/')
print(test_image_fps)

#make predictions on specific images supplied by image_fp and if pneumonia
#positive return image with drawn bounding boxes and prediction probabilities 
def predict(image_fp, min_conf):

    for idx,image_id in enumerate(image_fp):
        #ds = pydicom.read_file(image_id)
        #image = ds.pixel_array
        image = cv2.imread(image_id)
        # If grayscale. Convert to RGB for consistency.
        resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
		
        print(config.IMAGE_SHAPE[0])

        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1) 
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)
                
        patient_id = os.path.splitext(os.path.basename(image_id))[0]

        print(patient_id)

        results = model.detect([image])
        r = results[0]
        print(idx)
        print(r)
        
        for bbox,scores in zip(r['rois'],r['scores']): 
            print(bbox)
            x1 = int(bbox[1]) #* resize_factor)
            y1 = int(bbox[0]) #* resize_factor)
            x2 = int(bbox[3]) #* resize_factor)
            y2 = int(bbox[2])  #* resize_factor)
            print(x1," " ,y1," ",x2," ",y2)
            cv2.rectangle(image, (x1,y1), (x2,y2), (77, 255, 9), 3, 1)
            text = 'prediction = ' + str(scores)
            cv2.putText(image, text, (x1-50, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (77, 255, 9),lineType=cv2.LINE_AA,thickness=2)
            #cv2.imwrite(ROOTDIRECTORY + 'test_jpegs/pos_test339_labeled.jpg' ,image)
            width = x2 - x1 
            height = y2 - y1 
            print("x {} y {} h {} w {}".format(x1, y1, width, height)) 
            #plt.plot(image, cmap=plt.cm.gist_gray)
            #plt.subplot(1,2,(idx+1)) 
            #plt.plot(image, cmap=plt.cm.gist_gray)
            #plt.imshow(image, cmap=plt.cm.gist_gray)
            #cv2.imwrite(ROOTDIRECTORY + 'test_jpegs/pos_test339_labeled.jpg' ,image)
            #plt.show()
        #plt.figure()

        print(image_id)
        cv2.imwrite( os.path.splitext(image_id)[0] + '_labeled.jpg' ,image) 
        #plt.imshow(image, cmap=plt.cm.gist_gray)
        #plt.show()
        
predict(test_image_fps,.0005)











