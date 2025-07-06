from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss_GMM
from ssd_gmm import build_ssd_gmm
from utils.test_voc import *
from active_learning_loop import *
import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import math
import random
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from subset_sequential_sampler import SubsetSequentialSampler


random.seed(314) # need to change manually
torch.manual_seed(314) # need to change manually


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='VOC dataset root directory path')
parser.add_argument('--coco_root', default=COCO_ROOT,
                    help='COCO dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-3, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.5, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--eval_save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--use_cuda', default=True,
                    help='if True use GPU, otherwise use CPU')
parser.add_argument('--id', default=1, type=int,
                    help='the id of the experiment')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

cfg = voc300_active
    

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')    
    
net = build_ssd_gmm('test', cfg['min_dim'], cfg['num_classes'])
net = nn.DataParallel(net)
    
loop = 'weights/ssd300_AL_VOC_id_1_num_labels_5000_120000.pth'
net.load_state_dict(torch.load(loop))
net.eval()


import os.path as osp
VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")

test_dataset = VOCDetection(VOC_ROOT, [('2007', 'test')], BaseTransform(300, MEANS), VOCAnnotationTransform())
len(test_dataset)


num_images = len(test_dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
all_boxes = [[[] for _ in range(num_images)]
                for _ in range(len(labelmap)+1)]
                
im, gt, h, w = test_dataset.pull_item(1)
x = Variable(im.unsqueeze(0))




def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)) - 1,
                              int(float(bbox.find('ymin').text)) - 1,
                              int(float(bbox.find('xmax').text)) - 1,
                              int(float(bbox.find('ymax').text)) - 1]
        objects.append(obj_struct)

    return objects

annopath = os.path.join(VOC_ROOT, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(VOC_ROOT, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(VOC_ROOT, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = VOC_ROOT + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'
imagesetfile = imgsetpath.format(set_type)

with open(imagesetfile, 'r') as f:
    lines = f.readlines()
imagenames = [x.strip() for x in lines]
print(imagenames)

recs = {}
for i, imagename in enumerate(imagenames):
    recs[imagename] = parse_rec(annopath % (imagename))
    if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
            i + 1, len(imagenames)))