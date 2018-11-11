# YOLO Model

This is the YOLO model developed on a small dataset. A complete model is still training.

# File Instruction:
* `cfg` folder contains all the config files.
* `demo_images` folder contains 6 groups of test image, label and prediction result for both positive and negative case.
* `rsna_yolov3_100.weights` is the model weights after 100 iteration.
* `yolo_model_900.ipynb` is the Jupyter Notebook for developing this model over 900 training iterations.

# Test example
The ground truth image (left) and prediction (right) result:

<img src="demo_images/pos_test339_label.jpg" width="350" height="350" title="Ground Truth" hspace="20"/> <img src="demo_images/predictions_pos_test339.jpg" width="350" height="350" title="Prediction" hspace="20"/> 

<img src="demo_images/pos_test409_label.jpg" width="350" height="350" title="Ground Truth" hspace="20"/> <img src="demo_images/predictions_pos_test409.jpg" width="350" height="350" title="Prediction" hspace="20"/> 

<img src="demo_images/pos_test846_label.jpg" width="350" height="350" title="Ground Truth" hspace="20"/> <img src="demo_images/predictions_pos_test846.jpg" width="350" height="350" title="Prediction" hspace="20"/> 

<img src="demo_images/neg_test253_label.jpg" width="350" height="350" title="Ground Truth" hspace="20"/> <img src="demo_images/predictions_neg_test253.jpg" width="350" height="350" title="Prediction" hspace="20"/> 

<img src="demo_images/neg_test608_label.jpg" width="350" height="350" title="Ground Truth" hspace="20"/> <img src="demo_images/predictions_neg_test608.jpg" width="350" height="350" title="Prediction" hspace="20"/> 

<img src="demo_images/neg_test420_label.jpg" width="350" height="350" title="Ground Truth" hspace="20"/> <img src="demo_images/predictions_neg_test420.jpg" width="350" height="350" title="Prediction" hspace="20"/> 

As you can see, the result is not very good, but we are adding more data and training a more robust model.

