#!/bin/bash
cd /home/hshl/picar/SunFounder_PiCar-V/remote_control
python3 manage.py migrate
sudo ./start
