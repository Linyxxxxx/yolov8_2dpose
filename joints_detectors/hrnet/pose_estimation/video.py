'''
使用yolov3作为pose net模型的前处理
use yolov3 as the 2d human bbox detector
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
path1 = os.path.split(os.path.realpath(__file__))[0]
path2 = os.path.join(path1, '..')
sys.path.insert(0, path1)
sys.path.insert(0, path2)
import argparse
import pprint
import ipdb;pdb=ipdb.set_trace
import numpy as np
from tqdm import tqdm
from utilitys import plot_keypoint, PreProcess
import time

import torch
import _init_paths
from config import cfg
import config
from config import update_config

from lib.core.inference import get_final_preds
import cv2
import models
from scipy.signal import savgol_filter
sys.path.pop(0)
sys.path.pop(1)
sys.path.pop(2)


kpt_queue = []
def smooth_filter(kpts):
    if len(kpt_queue) < 6:
        kpt_queue.append(kpts)
        return kpts

    queue_length = len(kpt_queue)
    if queue_length == 50:
        kpt_queue.pop(0)
    kpt_queue.append(kpts)

    # transpose to shape (17, 2, num, 50) 关节点keypoints num、横纵坐标、每帧人数、帧数
    transKpts = np.array(kpt_queue).transpose(1,2,3,0)

    window_length = queue_length - 1 if queue_length % 2 == 0 else queue_length - 2
    # array, window_length(bigger is better), polyorder
    result = savgol_filter(transKpts, window_length, 3).transpose(3, 0, 1, 2) #shape(frame_num, human_num, 17, 2)

    # 返回倒数第几帧 return third from last frame
    return result[-3]


class get_args():
    # hrnet config
    cfg = path2 + '/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml'
    dataDir=''
    logDir=''
    modelDir=''
    opts=[]
    prevModelDir=''


##### load model
def model_load(config):
    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )
    model_file_name  = path2 + '/models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth'
    state_dict = torch.load(model_file_name, map_location=torch.device('cpu'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    return model

args = get_args()
update_config(cfg, args)


def get_pose_model():
    #  args = get_args()
    #  update_config(cfg, args)
    #### load pose-hrnet MODEL
    pose_model = model_load(cfg)
    pose_model.cuda()

    return pose_model


def get_pose(pose_model, image, bboxes):
    if len(bboxes)==0: return np.array([])
    
    # bbox is coordinate location
    inputs, origin_img, center, scale = PreProcess(image, bboxes, 1., cfg)

    with torch.no_grad():
        # compute output heatmap
        inputs = inputs[:,[2,1,0]]
        output = pose_model(inputs.cuda())
        # compute coordinate
        preds, maxvals = get_final_preds(
            cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

    # result = np.concatenate((preds[0], maxvals[0]), 1)
    result = np.concatenate((preds, maxvals), 2)

    return result

