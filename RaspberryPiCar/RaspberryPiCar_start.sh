#!/bin/bash

#   Shell script from: Joanna Rieger
#   > Shell script to start and stop all necessary processes for the RaspberryPiCar

cleanup() { 	                             
    echo "Cleaning up before exit."
    sudo killall -9 python3                    
} 

trap cleanup SIGINT SIGTERM


python3 /home/hshl/picar/SunFounder_PiCar-V/remote_control/remote_control/RaspberryPiCar_remote_car.py &
# 32-bit floating-point values
# python3 /home/hshl/tflite1/Rieger_TFLite_detection_webcam_send_Video.py --modeldir=/home/hshl/tflite1/custom_model_lite
# 8-bit integer values
# python3 /home/hshl/tflite1/Rieger_TFLite_detection_webcam_send_Video.py --modeldir=/home/hshl/tflite1/custom_model_lite --graph=detect_quant.tflite
# 8-bit integer values compiled to run on Edge TPU
python3 /home/hshl/tflite1/Rieger_TFLite_detection_webcam_send_Video.py --modeldir=/home/hshl/tflite1/custom_model_lite --edgetpu