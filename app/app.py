#!/usr/bin/env python

"""app.py: build a web application for pneumonia detection model using flask."""

__author__      = "minzhou"
__copyright__   = "Copyright 2018, minzhou@bu.edu"


import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from subprocess import call, Popen, PIPE
from shutil import copyfile
import cv2


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# predict and parse the output of yolo model
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
darknet_path = os.path.join(parent_path, 'yolo_model', 'darknet')
uploads_path = os.path.join(os.getcwd(), 'uploads')

def predict(input_image_path, threh=0.001):
    current_path = os.getcwd()
    os.chdir(darknet_path)
    p = Popen(['./darknet', 'detector', 'test', 
        '../cfg/rsna.data', '../cfg/rsna_yolov3.cfg_test', 
        '../backup/rsna_yolov3_900.weights', input_image_path, 
        '-thresh', f'{threh}'], stdout=PIPE)
    output = p.communicate()[0]
    os.chdir(current_path)
    return output

def parse_prediction_result(output):
    output = output[output.find('seconds.')+len('seconds.')+1:].replace('\n', '-').split('-')
    print(len(output))
    try:
        return(output[:5])
    except:
        return output

# Home
@app.route('/')
def index():
    return render_template('home.html')

# Upload
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    try:
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                invalidImage = 1
                return render_template('upload.html', invalidImage=invalidImage)
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                invalidImage = 1
                return render_template('upload.html', invalidImage=invalidImage)
            # success
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                input_image_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # predict using yolo model
                output = predict(input_image_path, 0.001)
                # output = parse_prediction_result(output)
                # copy the prediciton result to static folder
                prediction_path = os.path.join(darknet_path, 'predictions.jpg')
                oriimage = cv2.imread(prediction_path)
                newimage = cv2.resize(oriimage,(512,512))
                cv2.imwrite('static/predictions.jpg', newimage)                
                invalidImage = 2
                return render_template('upload.html', invalidImage=invalidImage, filename=filename, output=output)
            else:
                invalidImage = 1
                return render_template('upload.html', invalidImage=invalidImage)

        else:
            invalidImage = 3
            return render_template('upload.html', invalidImage=invalidImage)
    except:
        invalidImage = 3
        return render_template('upload.html', invalidImage=invalidImage)


# About
@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

# Contact
@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
     app.run(port = 5000, debug = True)
