# U-NET Chest X-Ray Lung Segmentation 

This Module contatins python scripts used to preprocess the RSNA Kaggle competition data, through the segmentation of the lungs out of chest X-Rays in the train and test datasets.  It uses a pretrained U-NET model to do so. 

## Getting Started

The UNET model has already been trained to an accuracy of about 98% , with the weights being attatched in the repo in the file unet_lung_seg.hdf5.  If the user wishes to implement and further train the model stored in andrew-s-rsna-lung-segmentation.py, they must visit these kaggle links to download training data from the [Shenzen Hospital](https://www.kaggle.com/yoctoman/shcxr-lung-mask) and [Montgomery County](https://www.kaggle.com/kmader/pulmonary-chest-xray-abnormalities).  

### Prerequisites

Download and install the [RSNA Stage 2 chest X-Rays](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) to be processed. 

Reqired Programs and Python Libaries (once python three is downloaded most can be downloaded using ```pip3 install program``` command

```
python/3.6.2 
tensorflow/r1.10 
shuntil 
openCV
numpy 
matplotlib 
pydicom 
png 
itertools
resizeimage
glob 
tqdm

```

### Installing

The data preprocessing is compleated in the file andrew-create-new-data_SCC.py. Before running the script ensure the paths to data are correct for local machine. I.E alter the code shown below to correct folders, two for the original DICOM images, two for intermediary png conversion folders, and two for the final .png images 

```
RSNA_TEST_IMG = "A1_Pneumonia_Detection_DATA/stage_2_test_images/"
RSNA_TRAIN_IMG = "A1_Pneumonia_Detection_DATA/stage_2_train_images/"
SEG_TRAIN_DIRECTORY = "A1_Pneumonia_Detection_PPDATA/stage_2_train_images_png/"
SEG_TEST_DIRECTORY = "A1_Pneumonia_Detection_PPDATA/stage_2_test_images_png/"
SEG_TRAIN_FINAL = "A1_Pneumonia_Detection_PPDATA/SEG_TRAIN/"
SEG_TEST_FINAL = "A1_Pneumonia_Detection_PPDATA/SEG_TEST/"
```

## How it works 

The scipt first converts the images from DICOM medical format to .PNG for ease of use int the Dicom_to_png() function. Then it loads the model architecture and creates the necessary helper functions to load the image, generate image bataches, and save the results. Finally it runs the pretrained model in prediction mode and saves the resulting segmented chest x-ray pngs in (512,512) .jpeg images 

## Examples 

Below are some example of the script working, I.E the input .png image, the generated mask, and the resulting saved segmented image. 

Original Image

![Original Image](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/Lung_Segmentation/Seg_Examples/2cf52eb6-785e-48e5-ae35-6b40da8d024e.png)

Generated Mask 

![Generated Mask](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/Lung_Segmentation/Seg_Examples/%20.png)

Segmented X-Ray

![Resulting Segmented X-Ray](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/Lung_Segmentation/Seg_Examples/2cf52eb6-785e-48e5-ae35-6b40da8d024e_predict%204.png)


## Authors

* **Andrew Stoycos** - astoycos@bu.edu

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.


## Acknowledgments

* Eduardo Mineo's kaggle kernal was used to train model -> (https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen)



