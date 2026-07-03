import os
import sys

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
from tools.utils.utils import *
import yaml



def read_one_need_from_seq(data_root_folder, file_num, seq_num):
    depth_data = \
        np.array(cv2.imread(data_root_folder + seq_num + "/depth_map/" + file_num + ".png",
                            cv2.IMREAD_GRAYSCALE))

    # depth_data = \
    #     np.array(np.load(data_root_folder + seq_num + "/depth_map/" + file_num + ".npy",
    #                      ))

    # depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor


### 
def read_one_need_from_seq_nclt(data_root_folder, file_num,seq_num):
    depth_data = np.load(data_root_folder+file_num+".npy")

    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor




def read_one_batch_pos_neg(data_root_folder, f1_index, f1_seq, train_imgf1, train_imgf2, train_dir1, train_dir2,
                           train_overlap, overlap_thresh):  

    batch_size = 0
    for tt in range(len(train_imgf1)):
        if f1_index == train_imgf1[tt] and f1_seq == train_dir1[tt]:
            batch_size = batch_size + 1

    sample_batch = torch.from_numpy(np.zeros((batch_size, 1, 32, 900))).type(torch.FloatTensor)
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor)

    pos_idx = 0
    neg_idx = 0
    pos_num = 0
    neg_num = 0

    for j in range(len(train_imgf1)):
        pos_flag = False
        if f1_index == train_imgf1[j] and f1_seq == train_dir1[j]:
            if train_overlap[j] > overlap_thresh:
                pos_num = pos_num + 1
                pos_flag = True
            else:
                neg_num = neg_num + 1

            depth_data_r = \
                np.array(cv2.imread(data_root_folder + train_dir2[j] + "/depth_map/" + train_imgf2[j] + ".png",
                                    cv2.IMREAD_GRAYSCALE))

            # depth_data_r = \
            #     np.array(np.load(data_root_folder + train_dir2[j] + "/depth_map/" + train_imgf2[j] + ".npy",
            #                      ))

            # depth_data_tensor_r = torch.from_numpy(depth_data_r).type(torch.FloatTensor).cuda()
            depth_data_tensor_r = torch.from_numpy(depth_data_r).type(torch.FloatTensor)
            depth_data_tensor_r = torch.unsqueeze(depth_data_tensor_r, dim=0)

            if pos_flag:
                sample_batch[pos_idx, :, :, :] = depth_data_tensor_r
                # sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor)
                pos_idx = pos_idx + 1
            else:
                sample_batch[batch_size - neg_idx - 1, :, :, :] = depth_data_tensor_r
                # sample_truth[batch_size - neg_idx - 1, :] = torch.from_numpy(np.array(train_overlap[j])).type(
                #     torch.FloatTensor).cuda()
                sample_truth[batch_size - neg_idx - 1, :] = torch.from_numpy(np.array(train_overlap[j])).type(
                    torch.FloatTensor)
                neg_idx = neg_idx + 1

    return sample_batch, sample_truth, pos_num, neg_num

#### 
def read_one_batch_pos_neg_nclt(f1_index, f1_seq, train_imgf1, train_imgf2, train_dir1, train_dir2,
                                  train_overlap, overlap_thresh, ri_bev_root):  
    batch_size = 0
    for tt in range(len(train_imgf1)):
        if f1_index == train_imgf1[tt] and f1_seq == train_dir1[tt]:
            batch_size = batch_size + 1
    # sample_batch = torch.from_numpy(np.zeros((batch_size, 10, 32, 900))).type(torch.FloatTensor).cuda()
    sample_batch = torch.from_numpy(np.zeros((batch_size, 10, 32, 900))).type(torch.FloatTensor)
    # sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor).cuda()
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor)
    pos_idx = 0
    neg_idx = 0
    pos_num = 0
    neg_num = 0
    for j in range(len(train_imgf1)):
        pos_flag = False
        if f1_index == train_imgf1[j] and f1_seq==train_dir1[j]:
            if train_overlap[j]> overlap_thresh:
                pos_num = pos_num + 1
                pos_flag = True
            else:
                neg_num = neg_num + 1
            depth_bev_data_r = np.load(ri_bev_root+train_imgf2[j]+".npy")
            
            # depth_bev_data_tensor_r = torch.from_numpy(depth_bev_data_r).type(torch.FloatTensor).cuda()
            depth_bev_data_tensor_r = torch.from_numpy(depth_bev_data_r).type(torch.FloatTensor)
            if pos_flag:
                sample_batch[pos_idx,:,:,:] = depth_bev_data_tensor_r
                # sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor)
                pos_idx = pos_idx + 1
            else:
                sample_batch[batch_size-neg_idx-1, :, :, :] = depth_bev_data_tensor_r
                # sample_truth[batch_size-neg_idx-1, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                sample_truth[batch_size-neg_idx-1, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor)
                neg_idx = neg_idx + 1
    return sample_batch, sample_truth, pos_num, neg_num



