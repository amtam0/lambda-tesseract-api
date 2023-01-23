import json
import base64
from PIL import Image, ImageFilter
import pytesseract as pt
import os
import numpy as np
import paddle_onnx_det_rec_EN
from io import BytesIO
import time

def tess_image_todata(CONTENT, config=None):
    """
    """
    data = pt.image_to_data(CONTENT, config=config, output_type=pt.Output.DICT)
    if data["text"]:
        emptywords_indices = [idx for idx, word in enumerate(data["text"]) if word.strip()==""]
        words = [el for idx, el in enumerate(data["text"]) if idx not in emptywords_indices]
        text = " ".join(words).strip()
        preds = [el for idx, el in enumerate(data["conf"]) if idx not in emptywords_indices]
    return [np.mean(preds),text]

def TESSERACT(CONTENT):
    """
    """
    SCORE, TEXT = tess_image_todata(CONTENT, config="--psm 6 -l eng")
    if not TEXT.strip():
        SCORE, TEXT = tess_image_todata(CONTENT, config="--psm 12 -l fra")
    return [SCORE,TEXT,"tesseract"]

def PADDLE_REC(CONTENT):
    """
    """
    if isinstance(CONTENT, list):
        results, results_info = ocr_sys.recognition_pil(CONTENT)
        return [el[0] for el in results]
    else:
        results, results_info = ocr_sys.recognition_pil([CONTENT])
        return [results[0][1], results[0][0], "paddle"]

def hybrid_ocr(pil_img=None, paddle_thresh=0.85):
    """
    """
    score, text, tool = PADDLE_REC(pil_img)
    if np.isnan(score):
        score = 0
    if score<paddle_thresh or len(text.strip())<=1:
        score, text, tool = TESSERACT(pil_img)
    return [score, text, tool]

def base64_to_pil(base64str):
    pil_image = Image.open(BytesIO(base64.b64decode(base64str)))
    pil_image = pil_image.convert('RGB')
    return pil_image

emp_img = Image.new("RGB", (100,40))
global ocr_sys
model_rec = "en_PP-OCRv3_rec_infer"
ocr_sys = paddle_onnx_det_rec_EN.det_rec_functions(np.array(emp_img),
                det_model='/opt/en_PP-OCRv3_det_infer.onnx',
                rec_model='/opt/{}.onnx'.format(model_rec),#en_PP-OCRv3_rec_infer
                en_dict='/opt/en_dict.txt')

def lambda_handler(event, context):
    
    # Extract content
    body_image64 = event['image64']
    ocrtool = event["ocrtool"] #"paddle"/"tesseract"/"both"
    paddle_thresh = event["paddle_thresh"]

    # Decode & open Img
    im = base64_to_pil(body_image64)

    # OCR
    if ocrtool=="tesseract":
        ocr_text = TESSERACT(im)
    elif ocrtool=="paddle":
        ocr_text = PADDLE_REC(im)
    elif ocrtool=="both":
        ocr_text = hybrid_ocr(im, paddle_thresh)

    event["ocr_results"] = str(ocr_text)
    del event["image64"]
    return {
        'statusCode': 200,
        'body': json.dumps(event),
        "layer_position":os.listdir("/opt")#optional
    }