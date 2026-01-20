# Drowsiness Detection using Deep Learning

This project implements a real-time drowsiness detection system using Deep Learning and Computer Vision techniques. 
A Convolutional Neural Network (CNN) is trained to classify eye states as open or closed. 
The system monitors live webcam input and triggers an audio alarm when prolonged eye closure is detected, helping to prevent accidents caused by fatigue.

## Features
- Real-time webcam-based detection
- CNN-based eye state classification
- Face detection to prevent false alarms
- Time-based drowsiness detection
- Audio alarm alert system

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy

## How to Run
```bash
pip install -r requirements.txt
python detect_drowsiness.py
