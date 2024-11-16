import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import argparse
import xml.etree.ElementTree as ET
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
net = build_ssd_gmm('test', 300, 20)
net = nn.DataParallel(net)
net.load_state_dict(torch.load(args.trained_model))

# Set the model to evaluation mode
net.eval()

# Load the VOC test dataset
testset = VOCDetection(args.dataset_root, [('2007', 'test')], None, VOCAnnotationTransform())

# Directory to save detection results as XML files
xml_output_dir = '/data/mn918/AL-MDN_Fs_vgg_19class/Generate_Pseduo_label_xml'
os.makedirs(xml_output_dir, exist_ok=True)


start_time = time.time()
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

    # Process and save the detection results in an XML file
    xml_file_path = os.path.join(xml_output_dir, f'{image_name}.xml')

    root = ET.Element("annotation")

    folder = ET.SubElement(root, "folder")
    folder.text = "VOC2007"

    filename = ET.SubElement(root, "filename")
    filename.text = f'{image_name}.jpg'  # Assuming the images have .jpg extension

    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image.shape[1])

    height = ET.SubElement(size, "height")
    height.text = str(image.shape[0])

    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.7:
            score = detections[0, i, j, 0].item()
            label_name = VOC_CLASSES[i - 1]
            pt = (detections[0, i, j, 1:5] * torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).cpu().numpy()
            xmin, ymin, xmax, ymax = pt[0], pt[1], pt[2], pt[3]

            obj = ET.SubElement(root, "object")
            name = ET.SubElement(obj, "name")
            name.text = label_name

            confidence = ET.SubElement(obj, "confidence")
            confidence.text = str(score)

            bbox = ET.SubElement(obj, "bndbox")
            xmin_elem = ET.SubElement(bbox, "xmin")
            xmin_elem.text = str(int(xmin))
            ymin_elem = ET.SubElement(bbox, "ymin")
            ymin_elem.text = str(int(ymin))
            xmax_elem = ET.SubElement(bbox, "xmax")
            xmax_elem.text = str(int(xmax))
            ymax_elem = ET.SubElement(bbox, "ymax")
            ymax_elem.text = str(int(ymax))

            j += 1

    tree = ET.ElementTree(root)
    tree.write(xml_file_path)

end_time = time.time()
# Calculate and print elapsed time
elapsed_time = end_time - start_time
print(f'Total execution time: {elapsed_time} seconds')

print(f'Detection results saved in Pascal VOC format to {xml_output_dir}')
