"""
######## Webcam Object Detection Using Tensorflow-trained Classifier #########
* Author: Evan Juras
* Date: 10/27/19
* Description: 
    This program uses a TensorFlow Lite model to perform object detection on a live webcam
    feed. It draws boxes and scores around the objects of interest in each frame from the
    webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
    This script will work with either a Picamera or regular USB webcam.

This code is based off the TensorFlow Lite image classification example at:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py

I added my own method of drawing boxes and labels using OpenCV. 
##############################################################################
.. py:module::      : Rieger_TFLite_detection_webcam_send_Video 

* Original program from: Evan Juras
        * Updated from: Joanna Rieger 
                * Updated to meet the requirements of the RaspberryPiCar
        
**********************************************************************

    * Filename      : Rieger_TFLite_detection_webcam_send_Video.py
    * Description   : A module for object recognition of road signs using 
                       a webcam and for sending the data as a live stream
    * Author        : Evan Juras 
        * Updated from:   Joanna Rieger
    * Project work  : "Entwicklung eines Modells zur Objekterkennung von 
                       Straßenschildern für ein ferngesteuertes Roboterauto"
    * E-mail        : joanna.rieger@stud.hshl.de
    
**********************************************************************
"""

################################################################################
#################################    SOURCES    ################################
################################################################################
# https://docs.sunfounder.com/_/downloads/picar-v/en/latest/pdf/
# https://github.com/sunfounder/SunFounder_PiCar-V/tree/V3.0

# https://www.youtube.com/watch?v=v0ssiOY6cfg&t=401s
# https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb
# https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/deploy_guides/Raspberry_Pi_Guide.md
# https://github.com/HumanSignal/labelImg/releases
# https://coral.ai/docs/accelerator/get-started/#3-run-a-model-on-the-edge-tpu

# https://zeromq.org/
# https://github.com/jeffbass/imagezmq


################################################################################
#################################    IMPORTS    ################################
################################################################################
import os
import argparse
import cv2
import numpy as np
import sys
import time
import importlib.util
import threading
import imagezmq
from threading import Thread
from multiprocessing import context


#################################################################################
#################################     CLASSES     ###############################
#################################################################################
class VideoStream:
    """
    Handles the sending of the video from the webcam in a separate processing 
    thread.
    
    * Source: Adrian Rosebrock, 
        PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/ 
    """
    
    def __init__(self, resolution=(320,240), framerate=30, video_port="6665"):  
        """
        Sets up the Camera
        
        Args:
            resolution (Any): defines the resolution for the camera images
            framerate (int): defines the frame rate
            video_port (str): defines the port for sending the image arrays
        """
        
        #----------------------------------------------------------------------
        # > Initialize the PiCamera and the camera image stream
        #----------------------------------------------------------------------
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        
        #----------------------------------------------------------------------    
        # > Read first frame from the stream
        #----------------------------------------------------------------------
        (self.grabbed, self.frame) = self.stream.read()
        
        #----------------------------------------------------------------------
        # > Create a thread for video sending
        #----------------------------------------------------------------------
        threading.Thread(target=self.send_video, args=(video_port,)).start()

        #----------------------------------------------------------------------
	    # > Variable to control when the camera is stopped
        #----------------------------------------------------------------------
        self.stopped = False

    def start(self):
        """
        * Starts the thread that reads frames from the video stream
        """  
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        """
        * Keeps looping indefinitely until the thread is stopped
        """
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return
            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        """
        * Returns the most recent frame
        """
        return self.frame
    
    def send_video(self, port):
        """
        * Sets up the sending of the image-arrays
        * Sends the image-arrays to the corresponding PC
                * image-arrays are send in a while-loop to create a livestream 
                   in the GUI created on the corresponding PC  
                    
        Args:
            port (Any): defines the port for sending the image arrays
        """
        sender = imagezmq.ImageSender(connect_to=f"tcp://*:{port}", REQ_REP=False)                       
        rpi_name = "RaspberryPiCar"                                             
        time.sleep(2.0)                      
                                                    
        while True:                                                              
            t_start = time.perf_counter()                                                       
            image = img_array[0]
            sender.send_image(rpi_name, image) 
            t_end = time.perf_counter()                                             
            time.sleep(1/10 -(t_end-t_start)) 

    def stop(self):
        """
        * Indicates that the camera and thread should be stopped
        """
        self.stopped = True
        

#################################################################################
#################################     SET UP     ################################
#################################################################################  
#---------------------------------------------------------------------- 
# > Define and parse input arguments
#---------------------------------------------------------------------- 
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='320x240')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

#---------------------------------------------------------------------- 
# > Import TensorFlow libraries
# > If tflite_runtime is installed, import interpreter from tflite_runtime, 
#    else import from regular tensorflow
# > If using Coral Edge TPU, import the load_delegate library
#---------------------------------------------------------------------- 
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

#---------------------------------------------------------------------- 
# > If using Edge TPU, assign filename for Edge TPU model
#---------------------------------------------------------------------- 
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

#---------------------------------------------------------------------- 
# > Get path to current working directory
#---------------------------------------------------------------------- 
CWD_PATH = os.getcwd()

#---------------------------------------------------------------------- 
# > Path to .tflite file, which contains the model that is used for 
#    object detection
#---------------------------------------------------------------------- 
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

#---------------------------------------------------------------------- 
# > Path to label map file
#---------------------------------------------------------------------- 
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

#---------------------------------------------------------------------- 
# > Load the label map
#---------------------------------------------------------------------- 
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

#---------------------------------------------------------------------- 
# > Have to do a weird fix for label map if using the COCO 
#    "starter model" from
#       > https://www.tensorflow.org/lite/models/object_detection/overview
#       > First label is '???', which has to be removed.
#---------------------------------------------------------------------- 
if labels[0] == '???':
    del(labels[0])

#---------------------------------------------------------------------- 
# > Load the Tensorflow Lite model.
# > If using Edge TPU, use special load_delegate argument
#---------------------------------------------------------------------- 
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

#---------------------------------------------------------------------- 
# > Get model details
#---------------------------------------------------------------------- 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

#---------------------------------------------------------------------- 
# > Check output layer name to determine if this model was created 
#    with TF2 or TF1, because outputs are ordered differently for 
#    TF2 and TF1 models
#---------------------------------------------------------------------- 
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

#---------------------------------------------------------------------- 
# > Initialize frame rate calculation
#---------------------------------------------------------------------- 
frame_rate_calc = 1
freq = cv2.getTickFrequency()

#---------------------------------------------------------------------- 
# > Initialize the video stream
#---------------------------------------------------------------------- 
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

#---------------------------------------------------------------------- 
# > Set up an array to write the image content to
#---------------------------------------------------------------------- 
Manager = context._default_context.Manager
img_array = Manager().list(range(2))
rt_img = np.ones((320,240),np.uint8)     
img_array[0] = rt_img


################################################################################
#################################   MAIN LOOP    ###############################
################################################################################
while True:
    #----------------------------------------------------------------------
    # > Start timer (for calculating frame rate)
    #----------------------------------------------------------------------
    t1 = cv2.getTickCount()

    #----------------------------------------------------------------------
    # > Grab frame from video stream
    #----------------------------------------------------------------------
    frame1 = videostream.read()

    #----------------------------------------------------------------------
    # > Acquire frame and resize to expected shape [1xHxWx3]
    #----------------------------------------------------------------------
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    #----------------------------------------------------------------------
    # > Normalize pixel values if using a floating model 
    #    (i.e. if model is non-quantized)
    #----------------------------------------------------------------------
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    #----------------------------------------------------------------------
    # > Perform the actual detection by running the model with the 
    #    image as input
    #----------------------------------------------------------------------
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    #----------------------------------------------------------------------
    # > Retrieve detection results
    #----------------------------------------------------------------------
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    #----------------------------------------------------------------------
    # > Loop over all detections and draw detection box and label if 
    #    confidence is above minimum threshold
    #----------------------------------------------------------------------
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (32, 178, 170), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # Draw label text

    #----------------------------------------------------------------------
    # > Draw framerate in corner of frame
    #----------------------------------------------------------------------
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,127,36),2,cv2.LINE_AA)

    #----------------------------------------------------------------------
    # > Assign the image content to an array 
    #----------------------------------------------------------------------
    img_array[0] = frame

    #----------------------------------------------------------------------
    # > Calculate framerate
    #----------------------------------------------------------------------
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    #----------------------------------------------------------------------
    # > Press 'q' to quit
    #----------------------------------------------------------------------
    if cv2.waitKey(1) == ord('q'):
        break
    
#----------------------------------------------------------------------
# > Clean up
#----------------------------------------------------------------------
videostream.stop()