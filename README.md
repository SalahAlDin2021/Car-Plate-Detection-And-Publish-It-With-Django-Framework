# Car-Plate-Detection-And-Publish-It-With-Django-Framework
## Introduction

This project is a web-based application that allows users to upload an image of a car, and then detects and returns the car's license plate. This application is developed using Django for the backend, HTML/CSS/JavaScript for the frontend, and integrates a pre-trained model for the license plate detection.

## Technologies

- **Django**: A high-level Python Web framework that encourages rapid development and clean, pragmatic design.
- **OpenCV (cv2)**: An open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in commercial products.
- **NumPy**: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- **easyocr**: A ready-to-use OCR tool that recognizes about 80+ languages, numbers, and special characters.

## Model

The model used in this project is trained using the Darknet framework, an open source neural network framework written in C and CUDA that is fast, easy to install, and supports CPU and GPU computation.

The pre-trained model can be found here: [License Plate Number Detection](https://github.com/pragatiunna/License-Plate-Number-Detection/tree/main). You can download the model directly from this [link](https://drive.google.com/u/0/uc?id=1cktcL1TXXRJ5o6CxzIuR08hPEWbb8Kkx&export=download).

To train your custom objects, you can refer to the Darknet's official GitHub page: [How to train to detect your custom objects](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects).

## Setup

To run this project, follow these steps:

1. Clone this repository: `git clone https://github.com/<your_username>/Car_Plate_Detector.git`
2. Navigate to the project directory: `cd Car_Plate_Detector`
3. Install the dependencies: `pip install -r requirements.txt`
4. Run the server: `python manage.py runserver`

Now, you should be able to navigate to `http://localhost:8000/` in your web browser to see the application.
