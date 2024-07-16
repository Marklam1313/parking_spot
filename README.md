# parking_spot
Real-time parking spot detection system

https://github.com/user-attachments/assets/f73b15c7-efba-43a1-a9a7-ac8026bac7ca

## Project Overview
This project develops a real-time parking spot monitoring system using computer vision techniques and machine learning. The system analyzes video feeds to detect empty and occupied parking spots, aiming to improve parking management efficiency.

## Features
Real-time Spot Detection: Identifies empty and occupied spots in real time.
Video Processing: Processes video input to continuously monitor parking status.
Visual Feedback: Marks empty spots in green and occupied spots in red on the video feed.

## How It Works
Mask Creation: Define the parking spots using a mask image where each spot is marked using Inkscape
Spot Detection: The system uses connected components in the mask to identify individual parking spots.
Status Determination: Each parking spot is periodically checked for occupancy using a trained model that predicts whether a spot is empty based on the visual data.
Visual Output: The program updates the video feed with colored rectangles indicating the status of each parking spot.

Source: https://www.youtube.com/watch?v=F-884J2mnOY&t=1534s



