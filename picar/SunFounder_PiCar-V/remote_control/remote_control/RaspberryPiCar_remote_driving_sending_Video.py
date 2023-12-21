"""
.. py:module::  : RaspberryPiCar_remote_driving_sending_Video

* Filename      : RaspberryPiCar_remote_driving_sending_Video.py
* Description   : A module used to remotely control the RaspberryPiCar and to
                   enable the displaying of the camera data on the PC used  
                   for the remote control.  
* Author        : Joanna Rieger
* Project work  : "Entwicklung eines Modells zur Objekterkennung von 
                   Straßenschildern für ein ferngesteuertes Roboterauto"
* E-mail        : joanna.rieger@stud.hshl.de
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
import picar
import math
import time
import cv2
import socket
import imagezmq                                             
import zmq       
import msgpack
import threading
from picar import back_wheels, front_wheels
from picar_v_video_stream import Vilib
from driver import camera


#################################################################################
#################################     CLASSES     ###############################
#################################################################################
class Car:
    """
    * Gets the RaspberryPiCar ready for usage
    * Sends image-arrays to allow livestreaming
    * Allows reacting to the received data by converting it into information for 
        the remote controlling of the RaspberryPiCar
    """
    PERIOD_LENGTH = 0.05
    FPS = 10

    def __init__(self, db_file="/home/hshl/SunFounder_PiCar-V/remote_control/driver/config", port="5556", video_port="6665") -> None:
        """
        * Sets up the RaspberryPiCar
        * Starts the camera and creates a thread to send the image-data
        
        Args:
            db_file (str): path to the configuration file
            port (str): defines the port for receiving the data from the PC used to remotely control the RaspberryPiCar 
            video_port (str): defines the port for sending the image arrays
        """ 
        picar.setup()                                                               
        self.fw = front_wheels.Front_Wheels(debug=False, db=db_file)                
        self.bw = back_wheels.Back_Wheels(debug=False, db=db_file)                  
        self.cam = camera.Camera(debug=False,db=db_file)                            
        self.cam.ready()                                                            
        self.bw.ready()                                                             
        self.fw.ready()    
                                                              
        Vilib.camera_start(web_func = False)                                                   
        threading.Thread(target=self.send_video, args=(video_port,)).start()  
                      
        self.com = COM(port)                                                     
               
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
        rpi_name = socket.gethostname()                                             
        time.sleep(2.0)        
                                                    
        while True:                                                              
            t_start = time.perf_counter()                                                       
            image = Vilib.img_array[0]
            sender.send_image(rpi_name, image) 
            t_end = time.perf_counter()                                             
            time.sleep(1/self.FPS -(t_end-t_start))                               

    def drive(self):
        """
        * Uses the received data and converts it into the speed and  
            steering angle for the RaspberryPiCar
        """
        while True:
            t_start = time.perf_counter()                                           
            command = self.com.get_data()                                           
            if command:
                speed = command["speed"]                                            
                angle = command["angle"]                                            
                self.bw.speed = abs(speed)                                              
                if speed > 0:                                                   
                    self.bw.forward()                                                               
                elif speed == 0:
                    self.bw.stop()                                                 
                else:
                    self.bw.backward()                                              
                    
                if angle % math.pi == 0:                                                    
                    self.fw.turn_straight()                                         
                else:
                    self.fw.turn(angle)                                                   
            t_end = time.perf_counter()                                                             
            time.sleep(self.PERIOD_LENGTH - (t_end-t_start))                            
        
       
class COM:
    """
    * Receives the data from the PC used to remotely control the RaspberryPiCar 
    """
    def __init__(self, port) -> None:
        """
        * Sets up the receiving of the data

        Args:
            port (Any): defines the port for receiving the data from the PC used to remotely control the RaspberryPiCar 
        """                                               
        self.context = zmq.Context()                                                
        self.socket = self.context.socket(zmq.PAIR)                                 
        self.socket.bind(f"tcp://*:{port}")                                             
        self.data = None                                                      
        threading.Thread(target=self.listen).start()                                       
    
    def listen(self):   
        """ 
	    * Receives the data for the speed and steering angle of the 
            RaspberryPiCar
        """                                                               
        while True:
            self.data = msgpack.unpackb(self.socket.recv())  
            
    def get_data(self):
        """
        * getter method to to get the received data 
        """
        return self.data 
                                                                          

#################################################################################
############################   SCRIPT ENTRY POINT    ############################
#################################################################################
#---------------------------------------------------------------------- 
# > is only called when the file is started as a script, not when it 
#    is imported
#----------------------------------------------------------------------   
if __name__ =="__main__": 
    car = Car(port="5556", video_port="6665")
    threading.Thread(target=car.drive).start()   