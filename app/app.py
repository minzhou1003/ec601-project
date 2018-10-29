#!/usr/bin/env python

"""app.py: build a web application for pneumonia detection model using flask."""

__author__      = "minzhou"
__copyright__   = "Copyright 2018, minzhou@bu.edu"


import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/predict_api', methods=['POST'])
# def predict():
#      # Error checking
#      data = request.get_json(force=True)

#      # Convert JSON to numpy array
#      predict_request = [data['sl'],data['sw'],data['pl'],data['pw']]
#      predict_request = np.array(predict_request)

#      # Predict using the random forest model
#      y = random_forest_model.predict(predict_request)

#      # Return prediction
#      output = [y[0]]
#      return jsonify(results=output)

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
                invalidImage = 2
                return render_template('upload.html', invalidImage=invalidImage)
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
