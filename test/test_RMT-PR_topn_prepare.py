import os
import sys

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../modules/')

import matplotlib.pyplot as plt
import torch
import yaml
import numpy as np

from modules.RMT_PR import featureExtracter

from tools.read_samples import read_one_need_from_seq_nclt


np.set_printoptions(threshold=sys.maxsize)
from tqdm import tqdm
import faiss
from tools.utils.utils import *


def shift(tensor, dim, index):
    length = tensor.size(dim)
    shifted_tensor = torch.cat((tensor.narrow(dim, index, length - index),
                                tensor.narrow(dim, 0, index)), dim=dim)
    return shifted_tensor

def unshift(tensor, dim, index):
    length = tensor.size(dim)
    unshifted_tensor = torch.cat((tensor.narrow(dim, length - index, index),
                                  tensor.narrow(dim, 0, length - index)), dim=dim)
    return unshifted_tensor

class testHandler():
    def __init__(self, height=32, width=900, channels=1, norm_layer=None, use_transformer=False,
                 data_root_folder=None, data_root_folder_test=None, test_weights=None):
        super(testHandler, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.use_transformer = use_transformer
        self.data_root_folder = data_root_folder
        self.data_root_folder_test = data_root_folder_test

        self.amodel = featureExtracter(channels=self.channels, use_transformer=self.use_transformer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        self.parameters = self.amodel.parameters()

        self.test_weights = test_weights

    def eval(self):

        print("Resuming From ", self.test_weights)
        checkpoint = torch.load(self.test_weights, map_location= self.device)
        self.amodel.load_state_dict(checkpoint['state_dict'])

        range_image_paths_database = load_files(self.data_root_folder)
        print("scan number of database: ", len(range_image_paths_database))

        des_list = np.zeros((len(range_image_paths_database), 1024)) 
        for j in tqdm(range(0, len(range_image_paths_database))):
            f1_index = str(j).zfill(6)
            current_batch = read_one_need_from_seq_nclt(self.data_root_folder, f1_index,seq_num=0)   #[1,5,32,900]
            
            ############
            divd =450
            input_batch = current_batch
            input_batch_trans1 = input_batch[..., :divd]
            input_batch_trans2 = input_batch[..., divd:]
            input_batch_trans = torch.cat([input_batch_trans2,input_batch_trans1],dim=-1)
            # input_batch_trans.requires_grad_(True)
            ##############

            current_batch = current_batch.to(self.device)
            input_batch_trans = input_batch_trans.to(self.device)


            # print(current_batch.shape)
            self.amodel.eval()
            current_batch_des = self.amodel(current_batch, input_batch_trans)

            des_list[(j), :] = current_batch_des[0, :].cpu().detach().numpy()

        des_list = des_list.astype('float32')

        nlist = 1
        k = 50
        d = 1024
        quantizer = faiss.IndexFlatL2(d)

        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        assert not index.is_trained

        index.train(des_list)
        assert index.is_trained
        index.add(des_list)
        row_list = []

        range_image_paths_query = load_files(self.data_root_folder_test)
        print("scan number of query: ", len(range_image_paths_query))

        for i in range(0, len(range_image_paths_query), 5):
        # for i in range(0, 20880, 5):    #for               04-05


            i_index = str(i).zfill(6)
            current_batch = read_one_need_from_seq_nclt(self.data_root_folder_test, i_index,seq_num=0)  

            ############
            divd =450
            input_batch = current_batch
            input_batch_trans1 = input_batch[..., :divd]
            input_batch_trans2 = input_batch[..., divd:]
            input_batch_trans = torch.cat([input_batch_trans2,input_batch_trans1],dim=-1)
            # input_batch_trans.requires_grad_(True)
            ##############

            current_batch = current_batch.to(self.device)
            input_batch_trans = input_batch_trans.to(self.device)

            self.amodel.eval()
            global_des_add = self.amodel(current_batch, input_batch_trans)

            des_list_current = global_des_add[0, :].cpu().detach().numpy()

            D, I = index.search(des_list_current.reshape(1, -1), k)  # actual search

            for j in range(D.shape[1]):
                one_row = np.zeros((1, 3))
                one_row[:, 0] = i
                one_row[:, 1] = I[:, j]
                one_row[:, 2] = D[:, j]
                row_list.append(one_row)
                print("query:" + str(i) + "---->" + "database:" + str(I[:, j]) + "  " + str(D[:, j]))

        row_list_arr = np.array(row_list)
        dir_name = "test/" 
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        np.savez_compressed(dir_name + "predicted_des_L2_dis_bet_traj_forward", row_list_arr)


if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../config/config.yml'
    
    config = yaml.safe_load(open(config_filename))
    data_root_folder = config["data_root"]["data_root_folder"]
    data_root_folder_test = config["data_root"]["data_root_folder_test"]
    test_weights = config["data_root"]["test_weights"]
    # ============================================================================

    test_handler = testHandler(height=32, width=900, channels=1, norm_layer=None, use_transformer=False,
                               data_root_folder=data_root_folder, data_root_folder_test=data_root_folder_test,
                               test_weights=test_weights)

    test_handler.eval()
