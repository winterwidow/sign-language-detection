# sign-language-detection

A real-time system that detects sign language gestures from a webcam, translates them into text, and then converts the text to speech using Python, MediaPipe, and machine learning.

## **modules to import:**

- mediapipe -- works only on python 3.10 not later versions

- scikit-learn

- opencv-python

- tensorflow

- pyttsx3

- numpy

## **Working**

1. Video Capture using OpenCV thorugh system webcam
2. Hand Detection using MediaPipe Hands
3. Extract Hand Landmarks
4. Sign CLassification using ML models - Scikit-learn/Tensorflow
5. Convert to text 
6. Speak text with pyttsx3

## **Datasets**

[ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)