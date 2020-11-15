# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np
import  common.pcl_dataloader_helper as pdh
import torch
import common.pcl as pcl

class ChunkedGenerator(torch.utils.data.Dataset):
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, use_pcl=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)
    
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        # Initialize buffers
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1])).astype('float32')
        self.batch_2d = np.empty((batch_size, chunk_length + 2*pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1])).astype('float32')

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.use_pcl = use_pcl
        
        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state
    
    def next_epoch(self):
        #enabled = True
        #while enabled:
        self.start_idx, self.pairs = self.next_pairs()

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        b_i = idx
        
            #for b_i in range(self.start_idx, self.num_batches):
        chunks = self.pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
        for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
            start_2d = start_3d - self.pad - self.causal_shift
            end_2d = end_3d + self.pad - self.causal_shift

            # 2D poses
            seq_2d = self.poses_2d[seq_i]
            low_2d = max(start_2d, 0)
            high_2d = min(end_2d, seq_2d.shape[0])
            pad_left_2d = low_2d - start_2d
            pad_right_2d = end_2d - high_2d
            if pad_left_2d != 0 or pad_right_2d != 0:
                self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
            else:
                self.batch_2d[i] = seq_2d[low_2d:high_2d]

            if flip:
                # Flip 2D keypoints
                self.batch_2d[i, :, :, 0] *= -1
                self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]

            # 3D poses
            if self.poses_3d is not None:
                seq_3d = self.poses_3d[seq_i]
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
                if pad_left_3d != 0 or pad_right_3d != 0:
                    self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                else:
                    self.batch_3d[i] = seq_3d[low_3d:high_3d]

                if flip:
                    # Flip 3D joints
                    self.batch_3d[i, :, :, 0] *= -1
                    self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                            self.batch_3d[i, :, self.joints_right + self.joints_left]


            # if i == 0 or i == 1023 or i==200 or i==500 or i==750:
            #     pdh.plot_the_skeletons(self.batch_2d[i])
            #     pdh.plot_the_skeletons_3d(self.batch_3d[i])
            #     before_3d = np.copy(self.batch_3d[i])

            if self.use_pcl:
                """THIS IS THE PCL CODE!"""
                pose2d_pt = torch.from_numpy(self.batch_2d[i].astype('float32'))
                middle_index = pose2d_pt.shape[0]//2
                location_py = pose2d_pt[middle_index][0,:]
                # location_py = pose2d_pt[:,0,:]
                
                pose2d_px = (pose2d_pt + 1) / 2 * 1000
                middle_index = pose2d_pt.shape[0]//2
                pose2d_middle = pose2d_px[middle_index]
                
                scale = pdh.generate_gt_scales_from2d(pose2d_middle).unsqueeze(0)
                location = pose2d_middle[0,:].unsqueeze(0)

                Ks_px_orig = torch.FloatTensor([
                    [1.145e3, 0, 5.0e2],
                    [0, 1.145e3, 5.0e2],
                    [0,    0,   1]
                ]).unsqueeze(0)

                augment_camera = False
                if augment_camera:
                    Ks_px_new = Ks_px_orig.clone()
                    f_factor = 0.6666
                    Ks_px_new[:,0,0] *= f_factor
                    Ks_px_new[:,1,1] *= f_factor

                    Ks_px_orig = Ks_px_new


                P_virt2orig, R_virt2orig, K_virt = pcl.pcl_transforms(location, scale, Ks_px_orig,\
                    focal_at_image_plane=True, slant_compensation=True)
                
                bs = pose2d_px.shape[0]
                num_joints = pose2d_px.shape[1]
                ones = torch.ones([bs, num_joints, 1])
                h_canon_label_2d = torch.cat((pose2d_px, ones), dim=-1).unsqueeze(-1)
                P_orig2virt = torch.inverse(P_virt2orig)
                # prep sizes for bmm (bxNxM) where N and M are matrix dimensions
                P_orig2virt = P_orig2virt.unsqueeze(1).repeat(bs, num_joints, 1, 1)
                P_orig2virt = P_orig2virt.view(bs*num_joints, 3, 3)
                h_canon_label_2d = h_canon_label_2d.view(bs*num_joints, 3, 1)

                # transform homogenous labels to virtual homogeneous labels
                h_canon_virt_2d = torch.bmm(P_orig2virt, h_canon_label_2d)
                h_canon_virt_2d = h_canon_virt_2d.squeeze(-1).view(bs, num_joints, -1)

                # Convert from homogeneous coordinate by dividing x and y by z
                pose2d_virt = torch.div(h_canon_virt_2d[:,:,:-1], h_canon_virt_2d[:,:,-1].unsqueeze(-1))
                pose2d_pt_pcl = pose2d_virt * 2 -1 
                # temp = pose2d_pt_pcl[middle_index]
                """NEW"""
                add_back_scale = 0
                if add_back_scale:
                    max_scale = torch.max(scale)
                    new_scale = torch.FloatTensor([max_scale, max_scale]).unsqueeze(0)
                    pose2d_pt_pcl = (pose2d_pt_pcl * (new_scale / 1000)) + location_py
                self.batch_2d[i] = pose2d_pt_pcl.numpy()

                
                
                """3D TRANSFORMS!"""
                pose3d_pt = torch.from_numpy(self.batch_3d[i].astype('float32'))
                pose3d_pt[:,0] = 0 # 0 should be the hip joint
                R_orig2virt = torch.inverse(R_virt2orig)
                R_orig2virt = R_orig2virt.unsqueeze(1).repeat(1, num_joints, 1, 1) #Repeats along 2nd dimension 16 times and for each seq
                pose3d_pt = pose3d_pt.unsqueeze(3).view(1*num_joints, 3, 1)
                R_orig2virt = R_orig2virt.view(1*num_joints, 3, 3)
                pose3d_pt_pcl = torch.bmm(R_orig2virt, pose3d_pt)
                pose3d_pt_pcl = pose3d_pt_pcl.squeeze(-1).view(1, num_joints, 3)

                self.batch_3d[i] = pose3d_pt_pcl.numpy()
                
                # if i == 0 or i == 1023 or i==200 or i==500 or i==750:
                #     pdh.plot_the_skeletons_3d(self.batch_3d[i])
                #     pdh.plot_the_skeletons(self.batch_2d[i])
                #     pdh.plot_the_before_after_skeletons_3d(before_3d, self.batch_3d[i])


            # if not self.use_pcl:
            #     """# Code used for centering"""
            #     # temp = self.batch_2d[i]
            #     # middle_index = temp.shape[0]//2
            #     # location_py = temp[middle_index][0,:]
            #     # location_py = np.expand_dims(location_py, 0)
            #     # location_py = np.expand_dims(location_py, 0)
            #     # location_py = np.repeat(location_py, temp.shape[1], 1)
            #     # location_py = np.repeat(location_py, temp.shape[0], 0)
            #     # self.batch_2d[i] = self.batch_2d[i] - location_py

            #     """# Code used for scaling"""
            #     temp = torch.from_numpy(self.batch_2d[i]) #.cuda()
            #     middle_index = temp.shape[0]//2
            #     middle_pose = temp[middle_index]
            #     middle_pose = (middle_pose + 1) / 2 * 1000
            #     scale = pdh.generate_gt_scales_from2d(middle_pose).unsqueeze(0)
            #     max_scale = torch.max(scale)
            #     new_scale = torch.FloatTensor([max_scale, max_scale]).unsqueeze(0)
            #     temp = temp / (new_scale / 1000) / 2
            #     self.batch_2d[i] = temp.numpy() # remove .numpy()
                
                


            # if i == 0 or i == 1023 or i==200 or i==500 or i==750:
            #     pdh.plot_the_skeletons_3d(self.batch_3d[i])
            #     # pdh.plot_the_skeletons(self.batch_2d[i])
            #     pdh.plot_the_skeletons(self.batch_2d[i])
            #     # pdh.plot_the_before_after_skeletons_3d(before_3d, self.batch_3d[i])
            

            # Cameras
            if self.cameras is not None:
                self.batch_cam[i] = self.cameras[seq_i]
                if flip:
                    # Flip horizontal distortion coefficients
                    self.batch_cam[i, 2] *= -1
                    self.batch_cam[i, 7] *= -1

        if self.endless:
            self.state = (b_i + 1, pairs)
        if self.poses_3d is None and self.cameras is None:
            return None, None, self.batch_2d[:len(chunks)]
        elif self.poses_3d is not None and self.cameras is None:
            # return None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
            return self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
        elif self.poses_3d is None:
            return self.batch_cam[:len(chunks)], None, self.batch_2d[:len(chunks)]
        else:
            return self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
    
    # if self.endless:
    #     self.state = None
    # else:
    #     enabled = False
            

class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None, use_pcl=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.use_pcl = use_pcl
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
        
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count
    
    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d in zip_longest(self.cameras, self.poses_3d, self.poses_2d):
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            
            # pdh.plot_the_single_skeletons(seq_2d)
            # pdh.plot_the_single_skeletons_3d(seq_3d)

            if self.use_pcl:
                """INSERT PCL CODE HERE"""
                pose2d_pt = torch.from_numpy(seq_2d.astype('float32'))
                pose2d_px = (pose2d_pt + 1) / 2 * 1000
                # middle_index = pose2d_pt.shape[0]//2
                # pose2d_middle = pose2d_px[middle_index]
                # temp = pose2d_px[0,:,:]
                scale = pdh.generate_batch_gt_scales_from2d(pose2d_px)
                location = pose2d_px[:,0,:]
                Ks_px_orig = torch.FloatTensor([
                    [1.145e3, 0, 5.0e2],
                    [0, 1.145e3, 5.0e2],
                    [0,    0,   1]
                ]).unsqueeze(0).repeat(scale.shape[0],1,1)

                P_virt2orig, R_virt2orig, K_virt = pcl.pcl_transforms(location, scale, Ks_px_orig,\
                        focal_at_image_plane=True, slant_compensation=True)
                
                bs = pose2d_px.shape[0]
                num_joints = pose2d_px.shape[1]
                ones = torch.ones([bs, num_joints, 1])
                h_canon_label_2d = torch.cat((pose2d_px, ones), dim=-1).unsqueeze(-1)
                P_orig2virt = torch.inverse(P_virt2orig)
                # prep sizes for bmm (bxNxM) where N and M are matrix dimensions
                P_orig2virt = P_orig2virt.unsqueeze(1).repeat(1, num_joints, 1, 1)
                P_orig2virt = P_orig2virt.view(bs*num_joints, 3, 3)
                h_canon_label_2d = h_canon_label_2d.view(bs*num_joints, 3, 1)

                # transform homogenous labels to virtual homogeneous labels
                h_canon_virt_2d = torch.bmm(P_orig2virt, h_canon_label_2d)
                h_canon_virt_2d = h_canon_virt_2d.squeeze(-1).view(bs, num_joints, -1)

                # Convert from homogeneous coordinate by dividing x and y by z
                pose2d_virt = torch.div(h_canon_virt_2d[:,:,:-1], h_canon_virt_2d[:,:,-1].unsqueeze(-1))
                pose2d_pt_pcl = pose2d_virt * 2 -1 
                # temp = pose2d_pt_pcl[0]

                pose2d_pt_pcl = pose2d_pt_pcl.numpy() # RETURN THIS!

                pose3d_pt = torch.from_numpy(seq_3d.astype('float32'))
                pose3d_pt[:,0] = 0 # 0 should be the hip joint
                R_orig2virt = torch.inverse(R_virt2orig)
                R_orig2virt = R_orig2virt.unsqueeze(1).repeat(1, num_joints, 1, 1) #Repeats along 2nd dimension 16 times and for each seq
                pose3d_pt = pose3d_pt.unsqueeze(3).reshape(bs*num_joints, 3, 1)
                R_orig2virt = R_orig2virt.view(bs*num_joints, 3, 3)
                pose3d_pt_pcl = torch.bmm(R_orig2virt, pose3d_pt)
                pose3d_pt_pcl = pose3d_pt_pcl.squeeze(-1).view(bs, num_joints, 3)

                pose3d_pt_pcl = pose3d_pt_pcl.numpy() # RETURN THIS!

                # pdh.plot_the_single_skeletons(pose2d_pt_pcl)
                # pdh.plot_the_single_skeletons_3d(pose3d_pt_pcl)

                batch_3d = None if pose3d_pt_pcl is None else np.expand_dims(pose3d_pt_pcl, axis=0)
                batch_2d = np.expand_dims(np.pad(pose2d_pt_pcl,
                                ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                                'edge'), axis=0)

            else:
                batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
                batch_2d = np.expand_dims(np.pad(seq_2d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            
            if self.augment:
                # Append flipped version
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1
                
                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam, batch_3d, batch_2d