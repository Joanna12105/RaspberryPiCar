#!/bin/bash

# https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/deploy_guides/Raspberry_Pi_Guide.md 
#   Original shell script from: Evan Juras
#   Updated from: Joanna Rieger 
#       Updated to meet the requirements of the RaspberryPiCar
#       Depending on the configuration of the Raspberry Pi used, 
#        additional/different installations may be necessary 
#   > Shell script to automatically install OpenCV, TensorFlow and all the dependencies needed

# Get packages required for OpenCV
sudo apt-get -y install libjpeg-dev libtiff5-dev libpng-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get -y install libxvidcore-dev libx264-dev
sudo apt-get -y install libatlas-base-dev

# Install OpenCV
pip3 install opencv-python==4.6.0.66

# Install TensorFlow
pip3 install tensorflow
