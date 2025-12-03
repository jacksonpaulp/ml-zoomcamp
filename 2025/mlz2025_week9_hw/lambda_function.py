import onnxruntime as ort

import torch
import torchvision.models as models
from torchvision import transforms

from io import BytesIO
from urllib import request
import numpy as np

from PIL import Image

target_size = (200,200)

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# ImageNet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Simple transforms - just resize and normalize
preprocess = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

onnx_model_path = "hair_classifier_empty.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name


def lambda_handler(event, context):
    url = event['url']
    img = download_image(url)
    img = prepare_image(img, target_size)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    X = batch_t.cpu().numpy().astype(np.float32)
    session_run = session.run([output_name],{input_name: X})
    float_prediction = session_run[0][0]
    return float_prediction.tolist()
