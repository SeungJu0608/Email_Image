import os,io
import pandas as pd
import numpy as np
import cv2
import base64
import re
from matplotlib import pyplot as plt
from google.cloud import vision
from google.cloud import translate
from google.cloud.vision_v1 import types
from bs4 import BeautifulSoup
from PIL import Image

key_file = '/content/drive/MyDrive/4-1/시스템분석/algebraic-link-330905-e64a5ace6944.json'

FILE_NAME = 'img_tf.jpg'
FOLDER_PATH = '/content/drive/MyDrive'

''' 
Image data is included in email data which MIMEType is 'text/html
1. html_decode func : decode function that are included with base64 encoded html data
2. uri_decode func : decode function that are included as uri data
3. attached_decode func : decode function that are included as attachment with base64 encoded
'''
def html_decode(data, key_file):  # key_file == personal google api key file
    os.environ['GOOGLE _APPLICATION_CREDENTIALS'] = key_file
    client = vision.ImageAnnotatorClient.from_service_account_json(key_file)

    decodedBytes = base64.urlsafe_b64decode(data)
    # print(decodedBytes)
    soup = BeautifulSoup(decodedBytes, "html.parser")
    # print('soup',soup)
    body = soup.body
    img_data = body.find('img').get('src')
    img_byte = re.sub('data:image/png;base64,', '', img_data)
    decodedBytes2 = base64.b64decode(img_byte)
    # print('decodedBytes2 : ',decodedBytes2)
    return decodedBytes2


def uri_decode(data, key_file):
    os.environ['GOOGLE _APPLICATION_CREDENTIALS'] = key_file
    client = vision.ImageAnnotatorClient.from_service_account_json(key_file)

    decodedBytes = base64.urlsafe_b64decode(data)
    soup = BeautifulSoup(decodedBytes, "html.parser")

    try:
        imgs = soup.findAll('img')
        src = []
        for img in imgs:
            temp = img.attrs['src']
            temp_url = temp.split(" ")[0]
            src.append(temp_url)
    except:
        pass

    return src


def attached_decode(data, key_file):
    os.environ['GOOGLE _APPLICATION_CREDENTIALS'] = key_file
    client = vision.ImageAnnotatorClient.from_service_account_json(key_file)

    decodedBytes = base64.urlsafe_b64decode(data)
    return decodedBytes

'''
OCR functions
1. ocr_image func : process of OCR system using a image bytes data
2. ocr_uri func : process of OCR system using a uri data as input
'''
def ocr_image(png_img, key_file):
    image = types.Image(content=png_img)
    client = vision.ImageAnnotatorClient.from_service_account_json(key_file)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    df = pd.DataFrame(columns=['locale', 'description'])
    for text in texts:
        df = df.append(
            dict(
                locale=text.locale,
                description=text.description
            ),
            ignore_index=True
        )

    return df['description'][0]


def ocr_uri(src_list, key_file):
    vision_client = vision.ImageAnnotatorClient.from_service_account_json(key_file)
    translate_client = translate.Client.from_service_account_json(key_file)

    result = ''
    for uri in src_list:
        try:
            source = types.ImageSource(image_uri=uri)

            img = types.Image(source=source)

            text_detection_response = vision_client.text_detection(image=img)
            annotations = text_detection_response.text_annotations

            if len(annotations) > 0:
                text = annotations[0].description
            else:
                text = ""

            detect_language_response = translate_client.detect_language(text)
            src_input = detect_language_response["input"]

            if src_input:
                result += src_input
                result += '\n'

        except:
            pass

    return result