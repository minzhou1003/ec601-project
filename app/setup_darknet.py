#!/usr/bin/env python

"""setup_darknet.py: download and set up the darknet for YOLOv3 model."""

__author__      = "minzhou"
__copyright__   = "Copyright 2018, minzhou@bu.edu"


import os
from pathlib import Path
import requests
from git import Repo
from subprocess import call


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def main():
    project_path = Path(Path.cwd()).parent
    backup_path = os.path.join(project_path, 'yolo_model', 'backup')
    darknet_url = 'https://github.com/pjreddie/darknet.git'
    darknet_path = os.path.join(project_path, 'yolo_model', 'darknet')

    # clone the darknet
    if not os.path.exists(darknet_path):
        Repo.clone_from(darknet_url, darknet_path)
    else:
        print('The darknet folder is already exists.')

    # go to darknet folder and make
    if not os.path.exists(os.path.join(darknet_path, 'darknet')):
        os.chdir(darknet_path)
        print(Path.cwd())
        call(['make'])
    else:
        print('The Makefile is already compiled.')

    # create backup folder
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)
    else:
        print('The backup folder is already exists.')

    # go to backup folder
    os.chdir(backup_path)

    # download the trained weights

    if not os.path.exists(os.path.join(backup_path, 'rsna_yolov3_900.weights')):
        id = '1Ju6VjrthMLs-nAOFXevpndJk8ePQo7iw'
        print('Downloading trained weights...')
        download_file_from_google_drive(id, 'rsna_yolov3_900.weights')
        print(f'Successfully downloaded weights.')
    else:
        print('The weights already exists.')

    print('Current path:')
    print(Path.cwd())

    
if __name__ == "__main__":
    main()
