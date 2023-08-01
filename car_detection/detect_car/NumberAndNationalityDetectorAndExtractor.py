import os
import cv2
from PIL import Image
import requests
import easyocr
import numpy as np

def improve_image_quality(image_path):
    # Read the image file
    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to load image at {image_path}")
        return None

    # Resize the image
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

    # Perform histogram equalization
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Perform Gaussian blur for noise reduction
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Perform image sharpening
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return img

def get_plate2(image_path):
    # Read the image file
    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to load image at {image_path}")
        return None

    # convert input image to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ensure it's 8-bit
    gray = gray.astype('uint8')

    # Perform histogram equalization to improve contrast
    gray = cv2.equalizeHist(gray)

    # construct the full file path to the Haar cascade
    cascade_file = os.path.join('../haarcascades', 'haarcascade_russian_plate_number.xml')

    # read haarcascade for number plate detection
    cascade = cv2.CascadeClassifier(cascade_file)

    # Detect license number plates
    plates = cascade.detectMultiScale(gray, 1.1, 3)
    print('Number of detected license plates:', len(plates))
    # if no plates were found, return None
    if len(plates) == 0:
        return None
    # loop over all plates
    for (x, y, w, h) in plates:
        # draw bounding rectangle around the license number plate
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        gray_plates = gray[y:y + h, x:x + w]
        color_plates = img[y:y + h, x:x + w]
    # save number plate detected
    cv2.imwrite('Numberplate.jpg', gray_plates)

    # save the original color version of the number plate
    cv2.imwrite('origin_plate.jpg', color_plates)

    return color_plates


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh



def ocr_on_plate(cropped):
    # Use easyocr to recognize the text
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped)
    return result



def check_nationality(plate_image_path):

    # Read the image file
    plate_image = cv2.imread(plate_image_path)

    # Check if the image was properly loaded
    if plate_image is None:
        print(f"Failed to load image at {plate_image_path}")
        return None

    # Convert the image to HSV
    hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)

    # Define range for yellow color
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the HSV image to get only yellow colors
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # If the yellow color is detected in the image then it is Israeli
    if cv2.countNonZero(yellow_mask) > 0:
        return "Israel"
    else:
        return "Palestinian"



url2 = "https://betterdatascience.com/detect-license-plates-with-yolo/images/3.png"
# url2 = ""
response = requests.get(url2, stream=True)
img = Image.open(response.raw)
img.save('temp.jpg')
img=improve_image_quality('temp.jpg')
cv2.imwrite('temp2.jpg', img)
cropped_image=get_plate2('temp2.jpg')
if cropped_image is None:
    print("No plate found in the image.")
else:
    # Then apply this function before OCR
    # cropped_image = preprocess(cropped_image)
    cv2.imwrite('cropped.jpg', cropped_image)
    cropped_image = preprocess(cropped_image)
    cv2.imwrite('cropped2.jpg', cropped_image)
    print(easyocr.__version__)
    # Perform OCR on the detected plate
    result = ocr_on_plate(cropped_image)
    # print(result)
    for res in result:
        print(res[1])  # print recognized text
        print(check_nationality('origin_plate.jpg'))

def getCarPlate(image_path):
    img = improve_image_quality(image_path)
    cv2.imwrite('processedImage.jpg', img)
    cropped_image = get_plate2('processedImage.jpg')
    if cropped_image is None:
        return None
    else:
        # Then apply this function before OCR
        # cropped_image = preprocess(cropped_image)
        cv2.imwrite('cropped.jpg', cropped_image)
        cropped_image = preprocess(cropped_image)
        cv2.imwrite('cropped2.jpg', cropped_image)
        # Perform OCR on the detected plate
        result = ocr_on_plate(cropped_image)
        # print(result)
        for res in result:
            return res[1]  # print recognized text

def getCarNationality(image_path):
    nationality= check_nationality(image_path)
    return nationality