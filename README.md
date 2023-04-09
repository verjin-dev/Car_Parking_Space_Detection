# Car Parking Space Detection using YOLO
This is a computer vision project that uses YOLO (You Only Look Once) algorithm to detect the availability of parking spaces in a parking lot in real-time.

## Project Overview
The goal of this project is to develop a system that can detect the number of available parking spaces in real-time, which can be used to optimize parking lot management and improve user experience. The project consists of the following components:

* Dataset: A dataset of parking lot images with labeled parking spaces.
* Preprocessing: Normalization of the dataset to improve the accuracy and performance of the YOLO algorithm.
* Training: Training the YOLO model on the preprocessed dataset using a deep learning framework such as Darknet or TensorFlow.
* Testing and Evaluation: Evaluating the performance of the model using metrics such as mean average precision (mAP) and intersection over union (IoU).
* Real-time Detection: Deploying the model for real-time detection of parking spaces in a video stream or a camera feed.
## Installation
To run this project, you will need to install the following dependencies:

* Python 3
* OpenCV
* NumPy
* YOLO weights and 
* configuration files
### You can install OpenCV and NumPy using pip:

```bash
pip install opencv-python
pip install numpy
```
To download the YOLO weights and configuration files, please follow the instructions in this [link](https://github.com/AlexeyAB/darknet#pre-trained-models).

## Usage
To run the car parking space detection system, you can use the following command:

```python
python detect_parking.py --input [input video file or camera index] --output [output video file] 
```
For example:

```python
 python detect_parking.py --input 0 --output output.avi
```
This will detect parking spaces in the camera feed from camera index 0 and save the output video to a file named output.avi.

## Credits
This project was inspired by this article by Carlo Ricci.

## License
This project is licensed under the MIT License - see the LICENSE file for details.