# Chest X-Ray Pneumonia diagnosis and bounding box detection using a backbone MASK_RCNN architecture. 

This Module is an attempt to complete the [Kaggle RSNA Pneumonia dection challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge). Specifically it is the attempt to build an algorithm that can detect the visual signal for pneumonia in medical chest x-rays, and return either pneumonia positive or negative, and if positive also return bounding boxs around the affected areas. 

## Getting Started

The data used in the project was a library of 26684 chest x_rays provided by the RSNA(Radiological Society of North America), they are provided in the medical standart DICOM format. With and accompying .csv file, the images all tagged as pneumonia positive or negative, where the pneumonia postive images also have bounding box data around the areas of intrest. 
For this project [Matterport's implemention](https://github.com/matterport/Mask_RCNN) of Mask_RCNN was employed. It is built on Python3, Keras, and Tensorflow and can bused with either a ResNet50 or ResNet101 backbone. 


### Prerequisites

Download the RSNA Chest X-Ray image dataset via [this link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)

Clone the Mask_Rcnn implementation by running the following code in the project directory 

```
git clone https://github.com/matterport/Mask_RCNN.git
```
Python and keras are essential to this script and must be downloaded, specifically 
```
python/3.6.2 
tensorflow/r1.10 
```

The libraries must be installed, most can be done using ``` pip3 install program ```

```
Shuntil 
Numpy 
Cv2 
Pydicom 
Png 
Itertols 
Resize image 
PIL 
Pandas 
Glob 


```

## Some Initial Thoughts

Pneumonia is an extremely devestating and inpactful disease which affects millions around the world today, so the prospect of using neural networks to help diagnose it was very exciting. However as a machine learning newbie I had much ground to cover before I could tackle such a difficult probem. Therefore I started by building a simple neural network architecture simply to tag a image as pneumonia positive or, which can be found [here](https://github.com/astoycos/Mini_Project2). Ultimatley this simple model architecture did not work very well, stemming from the fact that the patterns in lung opacities which it was trying to identify were very subtle, ultimately requiring a "deeper" architecture. However, it did allow me to get experience with data prepreocessing in python and the basics of Neural Network design. Next I begin researcing the vaious state of the art neuralnetwork architectures exisinting. After reading numerous papers and kernal kaggles I found the Mask_RCNN implementation, created by MIT.  It is unique in the fact that is allows for advanced pixel level segemntation, rather that simple bonunding box creation as see in other architectues such as RCNN and Faster RCNN. Therefore, I decided to progress in the project using it as my primary neural network architecture. To assist with the data preprocessing I also had the idea to segment the lungs out of the chest X - rays before using them to train the model in order to prevent exposing it to an erroneous data. Although I did not end up using [this module](https://github.com/astoycos/ec601-project/tree/master/Lung_Segmentation) it has many other practical uses. 

## Early Model train Evaluations 

To begin I ran three training attempts with the preprocessed data (from our lung segmentation module), the regular data, and the data + pretrained COCO weights on the initial layers.  Initially these three tests were run for only 16 epocs using Matterport's Mask_RCNN implementation, a resnet50 backbone,256 * 256 image input, and some standard config settings used by this [Kaggle Kernal](https://www.kaggle.com/hmendonca/mask-rcnn-and-coco-transfer-learning-lb-0-155). These settings allowed me to run relitively quickly on the free expernal servers provided by kaggle, these initial results are shown below: 

### PreProcessed Data (Lung Segmentation) Final Loss = 2.17
![pp_data](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/epocs%3D16_LR%3D.00005_.0005_L%3D2.17_PPDATA.jpeg)

### Regular Data (No pretrained weights) Final Loss = 1.85
![data](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/epocs%3D16_LR%3D.00005_.0005_L%3D1.8461_DATA.jpeg)

### Regular Data (using COCO pretrained weights) Final Loss = 1.39 
![data_and_coco](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/epocs%3D16_LR%3D.00005_.0005_L%3D1.39_DATA_cocoweights.jpeg)

Although it is no suprise the pretrained COCO dataset weights helped minimize training loss I was interested to see that the [preprocessed data](https://github.com/astoycos/ec601-project/tree/master/Lung_Segmentation) did much worse. However, after looking though the preprocessed Dataset I began to see some chest xrays with too much segmentation as shown below. 

PUT IN SEG IMAGES 

From the Previous results it was clear that further training and hyperparameter tuning should be compleated using non-segmented data with pretrained coco weights. Also, rather than using just losses to evaluate the accuracy of the model, a formal submission csv was created and submitted in order to acquire an official score from kaggle. Th.046 e submission format was as follows, 

with my intial model scoring a pretty dismal score of .04635. 

For the next training attempt I began by boosting the resolution of the input images from 256 X 256 to 512 X 512, thinking that the added resolution may help the model distinguish between radiograph background and lung opacities 

## Authors

* **Andrew Stoycos** - astoycos@bu.edu

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## Acknowledgments

@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
* Tian Xia's Kaggle Kernal -> https://www.kaggle.com/drt2290078/mask-rcnn-sample-starter-code
* etc
