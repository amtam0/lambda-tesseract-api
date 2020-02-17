import json
import cv2
import base64
import numpy as np
import pytesseract
from PIL import Image

#TODO
#### DECODE IMAGE WITHOUT SAVING IT

def write_to_file(save_path, data):
  with open(save_path, "wb") as f:
    f.write(base64.b64decode(data))

    
def ocr(img,oem=None,psm=None, lang=None):
  """
  ocr images if first config do not work, test second config
  """
  firstconfig='--oem {} --psm {} -l {}'.format(oem,psm,lang)
  
  text = pytesseract.image_to_string(img, config=firstconfig).strip()
  
  return text
      
def lambda_handler(event, context):
    
    # Extract content from json body
    body_image64 = event['image64']
    oem = event["tess-params"]["oem"]
    psm = event["tess-params"]["psm"]
    lang = event["tess-params"]["lang"]
    
    # Write request body data into file
    write_to_file("/tmp/photo.jpg", body_image64)
    
    # Read the image
    image = cv2.imread("/tmp/photo.jpg")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Write grayscale image to /tmp
    cv2.imwrite("/tmp/gray.jpg", gray)
    
    # Convert grayscale image into utf-8 encoded base64
    with open("/tmp/gray.jpg", "rb") as imageFile:
      str = base64.b64encode(imageFile.read())
      encoded_img = str.decode("utf-8")
    
    # Return the data to API Gateway in base64.
    # API Gateway will handle the conversion back to binary.
    # Set content-type header as image/jpeg.
    
    texttt = ocr(gray,oem=oem,psm=psm,lang=lang)
    
    return {
      "isBase64Encoded": True,
      "statusCode": 200,
      #"headers": { "content-type": "image/jpeg"},
      "body": texttt,
      "image":encoded_img,
      "oem":oem,
      "psm":psm,
      "lang":lang
    }

