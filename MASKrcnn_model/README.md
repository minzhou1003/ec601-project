# Chest X-Ray Pneumonia diagnosis and bounding box detection using a backbone MASK_RCNN architecture. 

This Module is an attempt to complete the [Kaggle RSNA Pneumonia dection challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge). Specifically it is the attempt to build an algorithm that can detect the visual signal for pneumonia in medical chest x-rays, and return either pneumonia positive or negative, and if positive also return bounding boxs around the affected areas. 

## Getting Started

The data used in the project was a library of 26684 chest x_rays provided by the RSNA(Radiological Society of North America), they are provided in the medical standart DICOM format. With and accompying .csv file, the images all tagged as pneumonia positive or negative, where the pneumonia postive images also have bounding box data around the areas of intrest. 
For this project [Matterport's implemention](https://github.com/matterport/Mask_RCNN) of Mask_RCNN was employed. It is built on Python3, Keras, and Tensorflow and can bused with either a ResNet50 or ResNet101 backbone. 

## Matterport's Mask-RCNN implementation 

![info](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/Screen%20Shot%202018-12-10%20at%205.17.11%20PM.png)

![info2](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/Screen%20Shot%202018-12-10%20at%205.17.20%20PM.png)

### Prerequisites

Download the RSNA Chest X-Ray image dataset via [this link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)

Clone the Mask_Rcnn implementation by running the following code in the project directory 

```
git clone https://github.com/matterport/Mask_RCNN.git
```

Python and Tensorflow are essential to this script and must be downloaded, specifically 

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

## Modules 

### andrew_RSNA_project.py 

This module is responsible for training the Mask_RCNN model, it creates the necessary data generators and config classes used by Mask_RCNN module for triaining.  Also it keeps track of the training progress and ultimately generates the loss vs epoch figures shown below 

### andrew_eval_MASKrcnn.py 

This module loads a pretrained model and writes the Kaggle submission file for that model instance. It also displays some examples of the model's predictions on random test images. 

### Mask_RCNN_app_model.py

This module is used by the project application to make a prediction on a specific image using a pretrained model, and return either the same image or the labeled image if pneumonia positive along with the bounding box prediction probabilities

## Some Initial Thoughts

Pneumonia is an extremely devestating and inpactful disease which affects millions around the world today, so the prospect of using neural networks to help diagnose it was very exciting. However as a machine learning newbie I had much ground to cover before I could tackle such a difficult probem. Therefore I started by building a simple neural network architecture simply to tag a image as pneumonia positive or, which can be found [here](https://github.com/astoycos/Mini_Project2). This simple model architecture did not work very well, stemming from the fact that the patterns in lung opacities which it was trying to identify were very subtle, ultimately requiring a "deeper" architecture. However, it did allow me to get experience with data prepreocessing in python and the basics of Neural Network design. Next I begin researcing the vaious state of the art neuralnetwork architectures exisinting. After reading numerous papers and kernal kaggles I found the Mask_RCNN implementation, created by MIT.  It is unique in the fact that is allows for advanced pixel level segemntation, rather that simple bonunding box creation as see in other architectues such as RCNN and Faster RCNN. Therefore, I decided to progress in the project using it as my primary neural network architecture. To assist with the data preprocessing I also had the idea to segment the lungs out of the chest X - rays before using them to train the model in order to prevent exposing it to an erroneous data. Although I did not end up using [this module](https://github.com/astoycos/ec601-project/tree/master/Lung_Segmentation) it has many other practical uses. 

## Early Model train Evaluations 

To begin I ran three training attempts with the preprocessed data (from our lung segmentation module), the regular data, and the data + pretrained COCO weights on the initial layers.  Initially these three tests were run for only 16 epocs using Matterport's Mask_RCNN implementation, a resnet50 backbone,256 * 256 image input, binary cross entropy loss functions and some standard config settings used by this [Kaggle Kernal](https://www.kaggle.com/hmendonca/mask-rcnn-and-coco-transfer-learning-lb-0-155). These settings allowed me to run relitively quickly on the free external servers provided by kaggle because Mask-RCNN would not work on my local machine, these initial results are shown below: 

### PreProcessed Data (Lung Segmentation) Final Loss = 2.17
![pp_data](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/epocs%3D16_LR%3D.00005_.0005_L%3D2.17_PPDATA.jpeg)

### Regular Data (No pretrained weights) Final Loss = 1.85
![data](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/epocs%3D16_LR%3D.00005_.0005_L%3D1.8461_DATA.jpeg)

### Regular Data (using COCO pretrained weights) Final Loss = 1.39 
![data_and_coco](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/epocs%3D16_LR%3D.00005_.0005_L%3D1.39_DATA_cocoweights.jpeg)

Although it is no suprise the pretrained COCO dataset weights helped minimize training loss I was interested to see that the [preprocessed data](https://github.com/astoycos/ec601-project/tree/master/Lung_Segmentation) did much worse. However, after looking though the preprocessed Dataset I began to see some chest xrays with too much segmentation as shown below. 

![segment_fail](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/Screen%20Shot%202018-12-09%20at%204.30.38%20PM.png)

From the Previous results it was clear that further training and hyperparameter tuning should be compleated using non-segmented data with pretrained coco weights. Also, rather than using just losses to evaluate the accuracy of the model, a formal submission csv was created and submitted in order to acquire an official score from kaggle. This score was calculated based on the IOU(intersection over union) of predicted vs actual bounding boxes. The submission format was as follows, 

![sub.csv](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/Screen%20Shot%202018-10-14%20at%2011.16.50%20PM.png)

with my intial model scoring a pretty dismal official score of .08425. 


## Moving forward in the training process 

As I began ramping up my training efforts, I moved from using Kaggle's free cloud resouces to Boston Univerity's shared computing cluster in order to have access to more computing resouces. While the hardware improved dramatically with the SCC, testing and iteration time also increased, often taking up to 3 days for a model to complete training. For the next few training attempts I did nothing but attempt to finetune the learning rate, I quickly setteled on one that varied in three steps.  For the first 5 epocs a larger LR of .001 along with the help of the pretrained COCO weights helped the model quckly identify features such as the edges and shapes of the lungs, then the LR was changed to .0005 for 5 epochs and finally .0001 for 6 epocs to minimize the viarability in the loss.  Although the official [MASK-RCNN paper](https://arxiv.org/abs/1703.06870) paper suggested a leraning rate of .02, I found that such a rate caused the weights to explode, essentially erasing further results. Also, when a smaller learing rate was tested, convergence time was extremely large requiring a large amount of epocs and usually resulting in overfitting. Following learning rate finetuning, I began boosting the resolution of the input images from 256 X 256 to 512 X 512, thinking that the added resolution may help the model distinguish between radiograph background and lung opacities. I addition I also bosted the # of epochs to 40 to see if the loss decline would continue to be constant. These changes resulted in much better outcomes, with a new lowest loss of 1.39 and updated Kaggle score of .11097. 

### Final Loss = 1.39 Kaggle score = .11097

![Loss=1.39](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/epocs%3D40_LR%3D.0005_L%3D1.39.jpeg)

Due to the large accurracy jumps with the above changes, I also exeperimented with changing the backbone neural network architecture to RESNET 101 and the input image size to 1024 X 1024, hoping that increased architecture complexity would help the model identify the minute opacity patterns. However, with both these changes the model complexity increased so much the so that even the powerful SCC hardware coulednt handle the burdon and results were meaningless. 

Thefore I took a step back and simply boosted the number of epocs to 100 to see if Loss could be further minmized without overshoot occuring.  This simple change allowed me to again achieve a better result with a Loss of 1.14 and Kaggle score of .13472

### Final Loss = 1.14 Kaggle score = .13472
![L=1.14](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/losses_vs_epocs-%3EL%3D1.14_100epocs.jpg)

## Current Training efforts  

As the end of the semester neared one of the fewe hyperparameters I had yet to touch was batch size, I.E the number of training images the model was exposed to in each epoch.  The original batch size was 200 so to see what would happen I increased it to a wopping 500 and boosted the epochs to 200.  With these dramatic changes I knew there was a high probablity of overfitting, since the model was being exposed to the training data so many times, and that is in fact what happened as shown below. 

![Overshoot](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/losses_vs_epocs_overshoot.png)

Therefore to minimize the overfitting problem I reduced the batch size to 300 and only trained for 100 epochs, which ultimately allowed me to achieve my best results to date, with a Loss = 1.0864 and Kaggle score of .13906.

### Final Loss = 1.0846 Kaggle score = .13906

![best](https://raw.githubusercontent.com/minzhou1003/ec601-project/master/MASKrcnn_model/Data/losses_vs_epocs-%3E.13906.png)


## Moving Forward 

This project is a work in progress, as machine learning is an extremely time consuming process each iterations can take weeks



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
* 
