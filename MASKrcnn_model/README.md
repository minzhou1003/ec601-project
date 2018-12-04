# Chest X-Ray Pneumonia diagnosis and bounding box detection using a backbone MASK_RCNN architecture. 

This Module is an attempt to complete the [Kaggle RSNA Pneumonia dection challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge). Specifically it is the attempt to build an algorithm that can detect the visual signal for pneumonia in medical chest x-rays. 

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

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Model Evaluations 

To begin I ran a three benchmark tests with the preprocessed data (from our lung segmentation module), the regular data, and the data + pretrained COCO weights on the initial layers.  Initially these three tests were run for only 16 epocs to get some benchmarks on how I should continue to tain in the future. 



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
