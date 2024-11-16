from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import BaseTransform
from data import VOC_CLASSES as labelmap
import torch.utils.data as data
import os
import sys
import time
import argparse
import numpy as np
import pickle
import cv2
import csv
from data import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT

from ssd_gmm import build_ssd_gmm

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model', default='weights/trained_VOC_name.pth', 
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.1, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=10, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='VOC dataset root directory path')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0.
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.1):
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = os.path.join(save_folder, 'ssd300')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detections_to_save = []
    unique_frame_id = 0

    for i in range(num_images):
        im, _, h, w = dataset.pull_item(i)
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(thresh).expand(15, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 15)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:5]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

            if len(cls_dets) > 0:
                image_id = dataset.ids[i][1]
                frame_id = unique_frame_id
                for idx, det in enumerate(cls_dets, start=1):
                    bbox = det[:4].tolist()
                    confidence = det[4]
                    class_name = labelmap[j - 1]
                    detections_to_save.append([idx, image_id, frame_id, *bbox, confidence, -1, class_name, confidence])

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))
        unique_frame_id += 1

    csv_data = [['# 1: Detection or Track-id', '2: Video or Image Identifier', '3: Unique Frame Identifier',
                 '4-7: Img-bbox(TL_x', 'TL_y', 'BR_x', 'BR_y)', '8: Detection or Length Confidence',
                 '9: Target Length (0 or -1 if invalid)', '10-11+: Repeated Species', 'Confidence Pairs or Attributes']]
    csv_data.extend(detections_to_save)

    with open(os.path.join(save_folder, 'detections.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(csv_data)

if __name__ == '__main__':
    num_classes = len(labelmap) + 1
    net = build_ssd_gmm('test', 300, num_classes)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    dataset = VOCDetection(args.dataset_root, [('2007', 'test')],
                           BaseTransform(300, (104, 117, 123)),
                           None)  # Remove VOCAnnotationTransform()
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(300, (104, 117, 123)), args.top_k, 300,
             thresh=args.confidence_threshold)
