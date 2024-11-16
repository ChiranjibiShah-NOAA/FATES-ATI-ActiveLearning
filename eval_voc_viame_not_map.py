import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
import csv
from ssd_gmm import build_ssd_gmm
from data import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector demo')
parser.add_argument('--trained_model', default='weights/trained_VOC_name.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='VOC dataset root directory path')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Initialize SSD
net = build_ssd_gmm('test', 300, 145)
net = nn.DataParallel(net)
net.load_state_dict(torch.load(args.trained_model))

# Set the model to evaluation mode
net.eval()

# Load the VOC test dataset
testset = VOCDetection(args.dataset_root, [('2007', 'test')], None, VOCAnnotationTransform())

# Output CSV file path
csv_file_path = 'detections.csv'

# Header for the CSV file
csv_header = ['# 1: Detection or Track-id', '2: Video or Image Identifier', '3: Unique Frame Identifier',
              '4-7: Img-bbox(TL_x', 'TL_y', 'BR_x', 'BR_y)', '8: Detection or Length Confidence',
              '9: Target Length (0 or -1 if invalid)', '10-11+: Repeated Species', 'Confidence Pairs or Attributes']

# Save detections to CSV
csv_data = [csv_header]

# Dictionary to map image names to unique frame identifiers
frame_id_dict = {}
for img_id in range(len(testset)):
    # Extract the image name from the dataset
    image_name = os.path.splitext(os.path.basename(testset.ids[img_id][1]))[0]
    
    # Get or create unique frame identifier
    unique_frame_id = frame_id_dict.get(image_name, len(frame_id_dict))
    frame_id_dict[image_name] = unique_frame_id

    # Form the correct path to the image
    image_path = os.path.join(args.dataset_root, 'VOC2007', 'JPEGImages', f'{image_name}.jpg')
    
    image = cv2.imread(image_path)

    # Preprocess the image
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = x.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        detections = net(xx)

    # Process and save the detection results
    detections_to_save = []  # Define detections_to_save here

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.5:
            score = detections[0, i, j, 0].item()  # Get the float value of the tensor
            label_name = VOC_CLASSES[i - 1]
            pt = (detections[0, i, j, 1:5] * torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).cpu().numpy()
            bbox = [pt[0], pt[1], pt[2], pt[3]]
            confidence = score
            class_name = label_name
            detections_to_save.append([j + 1, image_name, unique_frame_id, *bbox, confidence, -1, class_name, confidence])
            j += 1

    csv_data.extend(detections_to_save)

# Write CSV data to file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_data)

print(f'Detections saved to {csv_file_path}')
