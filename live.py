'''
This work is adapted from [AL-MDN](https://github.com/NVlabs/AL-MDN), and [SSD](https://github.com/lufficc/SSD/tree/master).
Please, check the README file for proper citation details. 

'''


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
import argparse
from ssd_gmm import build_ssd_gmm
from matplotlib import pyplot as plt
from data import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT, COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT
import torch
from torchvision import transforms


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector demo')
parser.add_argument('--trained_model', default='C:/Users/mn918/Active_learning/AL-MDN_Fs_vgg/weights/old_3/ssd300_AL_VOC_id_1_num_labels_10000_110000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='VOC dataset root directory path')
parser.add_argument('--coco_root', default=COCO_ROOT,
                    help='COCO dataset root directory path')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset == 'VOC':
    net = build_ssd_gmm('test', 300, 131)
else:
    net = build_ssd_gmm('test', 300, 131)
net = nn.DataParallel(net)
net.load_state_dict(torch.load(args.trained_model, map_location=device))
net = net.to(device)

video_path = 'C:/Users/mn918/Active_learning/ssd.pytorch/demo/761901372_cam4_5.mp4'  # Specify the path to your video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (300, 300)).astype(np.float32)
    image -= (104.0, 117.0, 123.0)
    image = image.astype(np.float32)
    image = image[:, :, ::-1].copy()
    image = torch.from_numpy(image).permute(2, 0, 1)
    image = image.unsqueeze(0).to(device)

    # Perform detection
    detections = net(image)
    # Process and visualize the detections as desired

    # Display the frame with detections
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


'''
from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse

import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from matplotlib import pyplot as plt
#from data import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT, COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='C:/Users/mn918/Active_learning/AL-MDN_Fs_vgg/weights/old_3/ssd300_AL_VOC_id_1_num_labels_10000_110000.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')                    
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net, transform):

    def transform_func(frame):
        transformed_frame = transform(frame)
        # Convert the transformed frame from a tensor to a NumPy array
        np_frame = transformed_frame.numpy()
        return np_frame
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform_func(frame)).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                            FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame


    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    #stream = WebcamVideoStream(src=0).start()  # default camera
    stream = cv2.VideoCapture('C:/Users/mn918/Active_learning/ssd.pytorch/demo/761901372_cam4_5.mp4') # mp4 video file

    time.sleep(1.0)
    fps = FPS().start()

    # Loop over frames from the video file stream
    while True:
        # Grab the next frame
        ret, frame = stream.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        # Update FPS counter
        fps.update()

        # Perform prediction on the frame
        frame = predict(frame)

        # Keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', frame)

        if key == 27:  # exit
            break

    # Stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # Cleanup
    cv2.destroyAllWindows()
    stream.release()



if __name__ == '__main__':
    import sys
    from os import path
    import torch
    import cv2
    from torchvision import transforms
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT, COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, BaseTransform
    from ssd_gmm import build_ssd_gmm

    if args.dataset == 'VOC':
        net = build_ssd_gmm('test', 300, 131)
    else:
        net = build_ssd_gmm('test', 300, 131)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(args.weights))    # initialize SSD

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((104/256.0, 117/256.0, 123/256.0), (1.0, 1.0, 1.0))
    ])

    fps = FPS().start()

    def predict(frame):
        im = transform(frame).unsqueeze(0).float()
        if args.cuda:
            im = im.cuda()
        detections = net(im).data

        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(15, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 15)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:5]
            boxes[:, 0] *= frame.shape[1]
            boxes[:, 2] *= frame.shape[1]
            boxes[:, 1] *= frame.shape[0]
            boxes[:, 3] *= frame.shape[0]
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)
            all_boxes[j][0] = cls_dets

        return frame

    cv2_demo(net.eval(), predict)  # Pass the predict function as an argument

    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()
    stream.stop()


if __name__ == '__main__':
    import sys
    from os import path
    from torchvision import transforms
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT, COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, BaseTransform
    from ssd_gmm import build_ssd_gmm
    
    if args.dataset == 'VOC':
        net = build_ssd_gmm('test', 300, 131)
    else:
        net = build_ssd_gmm('test', 300, 131)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(args.weights))    # initialize SSD

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((104/256.0, 117/256.0, 123/256.0), (1.0, 1.0, 1.0))
    ])

    fps = FPS().start()
    cv2_demo(net.eval(), transform)
    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()
    stream.stop()
    
    
'''    