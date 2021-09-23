# Detecting eye blinks

Hobby project that I started when my eyes got dry by sitting too much in front of the screen. 
I have been curios how much I blink per minute while working.

## Description
The code uses dlib 68 face landmark detection model to identify both eyes on the face. 
We grasp the points that are corresponding to the eyes and calculates the vertical aspect 
ratio which is the vertical distance of the upper and lower edge of the eyes. 
If this distance drops below a certain threshold value we count an eye blink.
You find the dlib model in the assets folder.
The following picture illustrates the dlib facial landmark coordinates:

<p align="center">
  <img src="/assets/68_facial_landmark_coordinates_dlib.jpg">
</p>

It is worth playing around a bit with the code to set the proper threshold value. Contours of both
eyes are marked on the video.
You can follow the video capture live, it is also saved to the disk if the corresponding variable 
(video_writing) is set to True.
Number of blinking per minute is saved to a csv file.

A very detailed tutorial and explanation of a blink detector can be found on 
[pyimagesearch](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
created by Adrian, many thanks for it.


## Getting Started
### Installing & get the code up and running

First clone the repository to your local machine or download it manually from git.

The requirements.txt contains all the neccessary packages, please make sure you install them. 
In order to do this, use your existing virtual env or create a new one, navigate to the root 
folder of the project and execute:

```
pip install -r requirements.txt
```

You can start the program by running the following module from the root directory:
```
python main.py
```
or you can run from your IDE through the same entry point, main.py.


