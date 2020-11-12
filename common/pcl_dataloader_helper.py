import numpy as np
import torch
import common.pcl as pcl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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

def plot_the_skeletons(skel_2d):
    # Plot 3 skeletons
    middle = skel_2d.shape[0]//2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    skel_0 = skel_2d[middle-10,:,:]
    skel_1 = skel_2d[middle-5,:,:]
    skel_2 = skel_2d[middle,:,:]
    skel_3 = skel_2d[middle+5,:,:]
    skel_4 = skel_2d[middle+10,:,:]
    plot_list = [skel_0, skel_1, skel_2, skel_3, skel_4]
    connections = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    for n, plot_skel in enumerate(plot_list):
        for i in range(1, plot_skel.shape[0]):
            if n < 2:
                color='blue'
            if n == 2:
                color='black'
            if n > 2:
                color='red'
            ax.plot([plot_skel[i,0], plot_skel[connections[i],0]], [plot_skel[i,1], plot_skel[connections[i],1]], color=color)
    plt.xlim(-1, 1)
    plt.ylim(1, -1)
    plt.show()

def plot_the_large_skeletons(skel_2d):
    # Plot 3 skeletons
    middle = skel_2d.shape[0]//2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    skel_0 = skel_2d[middle-10,:,:]
    skel_1 = skel_2d[middle-5,:,:]
    skel_2 = skel_2d[middle,:,:]
    skel_3 = skel_2d[middle+5,:,:]
    skel_4 = skel_2d[middle+10,:,:]
    plot_list = [skel_0, skel_1, skel_2, skel_3, skel_4]
    connections = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    for n, plot_skel in enumerate(plot_list):
        for i in range(1, plot_skel.shape[0]):
            if n < 2:
                color='blue'
            if n == 2:
                color='black'
            if n > 2:
                color='red'
            ax.plot([plot_skel[i,0], plot_skel[connections[i],0]], [plot_skel[i,1], plot_skel[connections[i],1]], color=color)
    plt.xlim(-2, 2)
    plt.ylim(2, -2)
    plt.show()

def plot_the_single_skeletons_3d(skel_3d):
    # Plot 3 skeletons
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_list = [skel_3d[0]]
    connections = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    for n, plot_skel in enumerate(plot_list):
        for i in range(1, plot_skel.shape[0]):
            if n == 0:
                color='blue'
            if n == 1:
                color='black'
            if n == 2:
                color='red'
            ax.plot([plot_skel[i,0], plot_skel[connections[i],0]], [plot_skel[i,1], plot_skel[connections[i],1]],\
                    [plot_skel[i,2], plot_skel[connections[i],2]],color=color)
    ax.set_xlim(-1, 1)
    
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=120, azim=-90)
    ax.set_ylim(1, -1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def plot_the_skeletons_3d(skel_3d):
    # Plot 3 skeletons
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    temp = skel_3d.squeeze(0)
    temp[0,:] = 0
    plot_list = [temp]
    connections = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    for n, plot_skel in enumerate(plot_list):
        for i in range(1, plot_skel.shape[0]):
            if n == 0:
                color='blue'
            if n == 1:
                color='black'
            if n == 2:
                color='red'
            ax.plot([plot_skel[i,0], plot_skel[connections[i],0]], [plot_skel[i,1], plot_skel[connections[i],1]],\
                    [plot_skel[i,2], plot_skel[connections[i],2]],color=color)
    ax.set_xlim(-1, 1)
    
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.view_init(elev=120, azim=-90)
    ax.set_ylim(1, -1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    
def plot_the_single_skeletons(skel_2d):
    # Plot 3 skeletons
    fig = plt.figure()
    ax = fig.add_subplot(111)
    skel_0 = skel_2d[0,:,:]

    plot_list = [skel_0]
    connections = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    for n, plot_skel in enumerate(plot_list):
        for i in range(1, plot_skel.shape[0]):
            if n < 2:
                color='blue'
            if n == 2:
                color='black'
            if n > 2:
                color='red'
            ax.plot([plot_skel[i,0], plot_skel[connections[i],0]], [plot_skel[i,1], plot_skel[connections[i],1]], color=color)
    plt.xlim(-1, 1)
    plt.ylim(1, -1)
    plt.show()
        

def plot_the_before_after_skeletons_3d(skel_3d, skel_3d_after):
    # Plot 3 skeletons
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    temp = skel_3d.squeeze(0)
    temp_1 = skel_3d_after.squeeze(0)
    # temp[0,:] = 0
    plot_list = [temp, temp_1]
    connections = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    for n, plot_skel in enumerate(plot_list):
        for i in range(1, plot_skel.shape[0]):
            if n == 0:
                color='black'
            if n == 1:
                color='red'
            ax.plot([plot_skel[i,0], plot_skel[connections[i],0]], [plot_skel[i,1], plot_skel[connections[i],1]],\
                    [plot_skel[i,2], plot_skel[connections[i],2]],color=color)
    
    ax.set_xlim(-1, 1)
    
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.view_init(elev=120, azim=-90)
    ax.set_ylim(1, -1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()