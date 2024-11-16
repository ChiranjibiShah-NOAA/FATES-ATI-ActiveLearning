import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
import csv
from ssd_gmm import build_ssd_gmm
from data import VOC_CLASSES

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector on video frames')
parser.add_argument('--trained_model', default='weights/trained_VOC_name.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--video_path', required=True, type=str, help='Path to the input video file or directory')
parser.add_argument('--frame_rate', default=5, type=int, help='Desired frames per second')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Initialize SSD
net = build_ssd_gmm('test', 300, 145)
net = nn.DataParallel(net)
net.load_state_dict(torch.load(args.trained_model))

# Set the model to evaluation mode
net.eval()

def extract_frames(video_file, output_dir, desired_fps):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error opening video file {video_file}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = -1  # Starting frame count
    extracted_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % int(fps / desired_fps) == 0:
            frame_identifier = frame_count // int(fps / desired_fps)
            filename = f"{os.path.splitext(os.path.basename(video_file))[0]}_{frame_identifier}.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame)
            extracted_frames.append((output_path, frame_identifier))
            print(f"Frame {frame_identifier} saved as {filename}")

    cap.release()
    return extracted_frames

# Define the directories
temp_dir = 'temp_image'
output_dir = 'output'

# Create temporary directory if not exists
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Function to process a single video
def process_video(video_path):
    # Extract frames from the video
    extracted_frames = extract_frames(video_path, temp_dir, args.frame_rate)

    if not extracted_frames:
        return

    # Output CSV file path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_file_path = os.path.join(output_dir, f'detection_{video_name}.csv')

    # Header for the CSV file
    csv_header = ['# 1: Detection or Track-id', '2: Video or Image Identifier', '3: Unique Frame Identifier',
                  '4-7: Img-bbox(TL_x', 'TL_y', 'BR_x', 'BR_y)', '8: Detection or Length Confidence',
                  '9: Target Length (0 or -1 if invalid)', '10-11+: Repeated Species', 'Confidence Pairs or Attributes']

    # Save detections to CSV
    csv_data = [csv_header]

    for frame_path, frame_id in extracted_frames:
        # Read the image
        image = cv2.imread(frame_path)

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
                bbox = [int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3])]  # Convert bbox to integer
                confidence = f"{score:.2f}"  # Format confidence to two decimal places
                class_name = label_name
                detections_to_save.append([j + 1, os.path.basename(frame_path), frame_id, *bbox, confidence, -1, class_name, confidence])
                j += 1

        csv_data.extend(detections_to_save)

    # Write CSV data to file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(csv_data)

    print(f'Detections saved to {csv_file_path}')

    # Clean up temporary frames directory
    for frame_path, _ in extracted_frames:
        os.remove(frame_path)

# Check if the video_path is a directory or a single video file
if os.path.isdir(args.video_path):
    video_files = [os.path.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video_file in video_files:
        process_video(video_file)
else:
    process_video(args.video_path)
