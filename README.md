# Football Analysis Project

## Introduction
This project leverages advanced AI and computer vision techniques to detect and track players, referees, and the ball in football games. It also provides insightful analytics such as player speed, distance covered, ball trajectory, and team possession statistics. 


## Features

- **Player Detection and Tracking**:
  Tracks individual players on the field and annotates them with unique IDs.
  
- **Ball Detection and Trajectory Analysis**:
  Detects the ball in real time and calculates its trajectory throughout the game.

- **Team Assignment**:
  Uses k-means clustering to segment and assign team colors automatically.

- **Optical Flow and Perspective Transformation**:
  Measures camera movement and represents scene depth to provide accurate annotations.

- **Player Speed and Distance Calculations**:
  Computes individual player speeds and distances covered during the match.

- **Team Ball Control Statistics**:
  Tracks which team has control of the ball and visualizes the percentage of ball possession.

## To Train model use 
- yolo task=detect mode=predict model=yolov8n.pt source=input_videos/video_file.mp4

## Modules Used

The following modules are used in this project:

- **YOLO (You Only Look Once)**: An advanced AI-based object detection model used to detect players, referees, and the ball in real-time from the video frames.
  
- **KMeans**: A clustering algorithm applied for pixel segmentation to detect and classify t-shirt colors, enabling the identification of players from different teams.

- **Team Assigner**: Automatically assigns players to their respective teams based on their detected t-shirt colors using KMeans clustering, ensuring accurate team tracking.

- **Optical Flow**: Utilized to measure camera movement, ensuring accurate tracking even when the camera shifts during gameplay.
  
- **Perspective Transformation**: A mathematical technique to represent scene depth and adjust the perspective, making it easier to analyze player positions and movements on the field.
  
- **Speed and Distance Calculation**: Calculates the speed and distance covered by each player during the game by analyzing frame-by-frame positional data from bounding boxes.

## Instructions to Run the Project

Follow the steps below to run the project:

### 1. Clone the Repository and Run 
Clone this repository to your local machine using the following command:
```bash
- git clone repository-url
- cd <project-folder>
- python main.py


## Requirements
To run this project, you need to have the following requirements installed:
- Python 
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas