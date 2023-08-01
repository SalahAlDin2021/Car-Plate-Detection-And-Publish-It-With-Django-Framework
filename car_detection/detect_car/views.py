import base64
from django.http import JsonResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2
import easyocr
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from car_detection.settings import BASE_DIR

# csrf_exempt is a decorator which can be applied to a view function
# where Cross Site Request Forgery protection should be bypassed.
@csrf_exempt
def upload_image(request):
    try:
        # Check if the request is a POST request
        if request.method == 'POST':
            # Retrieve the file from the request
            uploaded_file = request.FILES['document']
            # Initialize the FileSystemStorage
            fs = FileSystemStorage()
            file_path = 'detectPlateImages\\image.png'
            # Check if the file already exists and if so, delete it
            if fs.exists(file_path):  # Check if the file already exists
                fs.delete(file_path)  # If it does, delete it
            # Save the uploaded file to the specified location and retrieve the absolute path
            name = fs.save(file_path, uploaded_file)
            image_path = fs.path(file_path)
            # Use the getCarPlate function (which should be defined elsewhere) to process the image
            car_number = getCarPlate(image_path)
            # If the car number plate couldn't be detected
            if car_number is None:
                with open(image_path, "rb") as image_file:
                    # Convert your image to Base64 to send it as json object
                    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                # Return a JsonResponse with an error message and the image in base64 format
                return JsonResponse(
                    { 'nationality': "Sorry, The Car Plate doesn't Detected", 'image_base64': image_base64})

            # If the car number plate was detected, determine the nationality
            nationality = check_nationality('detectPlateImages\\plate.png')

            with open('detectPlateImages\\detected_car_plate.jpg', "rb") as image_file:
                # Convert your image to Base64
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            # Return a JsonResponse with the nationality and the image in base64 format
            # Return JsonResponse instead of rendering HTML
            return JsonResponse({'nationality': nationality, 'image_base64': image_base64})
        # If the request isn't a POST request, render the index page
        return render(request, 'detect_car/index.html')
    # If there was an exception, return that the car plate not found in the image.
    except Exception as e:
        with open(image_path, "rb") as image_file:
            # Convert your image to Base64
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # Return a JsonResponse with an error message and the image in base64 format
        return JsonResponse({ 'nationality': "Sorry, The Car Plate doesn't Detected", 'image_base64': image_base64})
















def improve_image_quality(image_path):
    '''
    This function enhances the quality of the input image by performing several operations such as resizing,
    histogram equalization, Gaussian blurring, and sharpening.
    '''
    # Read the image file using OpenCV
    img = cv2.imread(image_path)
    # Check if the image is loaded properly
    if img is None:
        print(f"Failed to load image at {image_path}")
        return None
    # Resize the image by 20% for better quality
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    # Convert the image to YUV format and equalize the histogram of the Y channel
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # Convert the image back to BGR format
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    # Apply Gaussian blur for noise reduction
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply image sharpening: Subtract the blurred image from the original image to get the sharpness mask
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return img

def get_plate(img, model, classes, conf_threshold=0.5, nms_threshold=0.4):
    '''
        This function detects and extracts the car plate from an image using the YOLOv3 model.
        '''
    # Get the image's height and width
    height, width = img.shape[:2]

    # Convert the image to blob for YOLOv3 input
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
    # Set the blob as input to the model
    model.setInput(blob)
    # Perform a forward pass through the network to get the output predictions
    outs = model.forward(model.getUnconnectedOutLayersNames())
    # this arrays to save the dimensions of detected car plate
    class_ids = []
    confidences = []
    boxes = []
    # For each detection from each output layer, get the confidence, class id, bounding box parameters
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                # Calculate coordinates based on the center of the bounding box and its width and height
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # Store the class id, confidence and bounding box coordinates
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Go through the detections remaining after nms and draw bounding box
    # For each detected object after non-max suppression
    for i in indices:
        i = i[0] if isinstance(i, list) or isinstance(i, np.ndarray) else i
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        # Clamp the box coordinates to make sure they don't exceed image boundaries
        # Make sure box is within image
        x = max(min(x, width - 1), 0)
        y = max(min(y, height - 1), 0)
        w = min(w, width - x)
        h = min(h, height - y)

        # Draw a green bounding box around the detected object
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 15)

        # Crop the object(car plate) from the image and save it
        cropped = img[y:y + h, x:x + w]
        cv2.imwrite("detectPlateImages\\plate.png".format(i), cropped)

    # Save the image with rectangle(detected car plate)
    cv2.imwrite("detectPlateImages\\detected_car_plate.jpg", img)
    return cropped

def preprocess(image):
    '''
    This function converts the image to grayscale, applies Gaussian blur, and then uses Otsu's method for binarization.
    '''
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Otsu's thresholding for binarization
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def ocr_on_plate(cropped):
    '''
        This function performs Optical Character Recognition (OCR) on the cropped plate image.
        '''
    # Initialize the OCR tool with the language that should read
    reader = easyocr.Reader(['en'])
    # Perform OCR on the image
    results = reader.readtext(cropped)

    # Extract and concatenate text from each result
    text = ''.join(result[1] for result in results)  # result[1] is the recognized text

    return text



def check_nationality(plate_image_path):
    '''
        This function determines the nationality of a car based on the plate color. If the plate is yellow, the car is from
        Israel, otherwise it is from Palestine.
        '''
    # Read the image file
    plate_image = cv2.imread(plate_image_path)

    # Check if the image was properly loaded
    if plate_image is None:
        print(f"Failed to load image at {plate_image_path}")
        return None

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)

    # Define range for yellow color
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask of the yellow regions in the image
    # Threshold the HSV image to get only yellow colors
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # If the yellow color is detected in the image then it is Israeli car plate
    if cv2.countNonZero(yellow_mask) > 0:
        return "Israel"
    else:
        return "Palestinian"


def getCarPlate(image_path):
    '''
    This function performs the entire process of car plate recognition.
    '''
    # the model downloaded from
    # https://github.com/pragatiunna/License-Plate-Number-Detection/tree/main
    # https://drive.google.com/u/0/uc?id=1cktcL1TXXRJ5o6CxzIuR08hPEWbb8Kkx&export=download

    # and this model traind using darknet, and below you can know how to train to detect your custom objects
    # https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
    # Load the classes names
    with open("haarcascades\\classes.names") as f:
        classes = [line.strip() for line in f.readlines()]

    # Set up the YOLOv3 model
    # Set up the neural network
    model = cv2.dnn.readNet("haarcascades\\lapi.weights", "haarcascades\\darknet-yolov3.cfg")
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # Enhance the image quality
    img = improve_image_quality(image_path)
    # Detect and extract the car plate from the image
    cropped = get_plate(img, model, classes)
    # Preprocess the cropped plate image for OCR
    cropped = preprocess(cropped)
    # Perform OCR on the plate image to get the text
    result = ocr_on_plate(cropped)
    return result