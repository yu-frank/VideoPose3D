import numpy as np
import torch
import common.pcl as pcl

def generate_gt_scales_from2d(pose_2d):
    max_y = torch.max(pose_2d[:,1])
    min_y = torch.min(pose_2d[:,1])
    max_x = torch.max(pose_2d[:,0])
    min_x = torch.min(pose_2d[:,0])
    scale_y = max_y - min_y
    scale_x = max_x - min_x
    # scale = torch.max(scale_y, scale_x)
    return torch.tensor([scale_x, scale_y])

def generate_batch_gt_scales_from2d(pose_2d):
    max_y, _ = torch.max(pose_2d[:,:,1], dim=1)
    min_y, _ = torch.min(pose_2d[:,:,1], dim=1)
    max_x, _ = torch.max(pose_2d[:,:,0], dim=1)
    min_x, _ = torch.min(pose_2d[:,:,0], dim=1)
    scale_y = max_y - min_y
    scale_x = max_x - min_x
    # scale = torch.max(scale_y, scale_x)
    return torch.stack([scale_x, scale_y], dim=1)
