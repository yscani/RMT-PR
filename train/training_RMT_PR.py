import os
import sys

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../modules/')
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter    
from modules.RMT_PR import featureExtracter
from tools.read_samples import read_one_batch_pos_neg_nclt
from tools.read_samples import read_one_need_from_seq_nclt

np.set_printoptions(threshold=sys.maxsize)
import modules.loss as PNV_loss
from tools.utils.utils import *
import yaml
from datetime import datetime


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


def imgf1_imgf2_overlap_nclt(rain_set_imgf1_imgf2_overlap, shuffle=True):

    if shuffle:
        train_set_imgf1_imgf2_overlap = np.random.permutation(rain_set_imgf1_imgf2_overlap)

        train_imgf1 = train_set_imgf1_imgf2_overlap[:, 0]
        train_imgf2 = train_set_imgf1_imgf2_overlap[:, 1]
        train_dir1 = np.zeros((len(train_imgf1),))
        train_dir2 = np.zeros((len(train_imgf2),))
        train_overlap = train_set_imgf1_imgf2_overlap[:, 2].astype(float)
  
    return (train_imgf1, train_imgf2, train_dir1, train_dir2, np.asarray(train_overlap))


class trainHandler():
    def __init__(self, height=32, width=900, channels=5, norm_layer=None, lr=0.001,
                 data_root_folder=None, train_set=None, training_seqs=None):
        super(trainHandler, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.learning_rate = lr
        self.data_root_folder = data_root_folder
        self.train_set = train_set
        self.training_seqs = training_seqs

        self.amodel = featureExtracter(channels=self.channels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       
        self.amodel.to(self.device)
        self.parameters = self.amodel.parameters()
        self.optimizer = torch.optim.Adam(self.parameters, self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)

        self.rain_set_imgf1_imgf2_overlap = np.load(train_set)    


        (self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.train_overlap) = \
            imgf1_imgf2_overlap_nclt(self.rain_set_imgf1_imgf2_overlap)

        """change the args for resuming training process"""
        self.resume = False
        os.makedirs("./checkpoints", exist_ok=True)
        self.save_name = "./checkpoints/trained_RMT_PR49.pth.tar"

        
        """overlap threshold for positive/negative sample mining"""
        self.overlap_thresh = 0.3

    def train(self):

        epochs = 50

        """resume from the saved model"""
        if self.resume:
            resume_filename = self.save_name
            print("Resuming from ", resume_filename)
            checkpoint = torch.load(resume_filename)
            starting_epoch = checkpoint['epoch']
            self.amodel.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("Training From Scratch ...")
            starting_epoch = 0

        writer1 = SummaryWriter(comment="LR_0.000005", log_dir="test_device")

        for i in range(starting_epoch + 1, epochs):



            (self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.train_overlap) = \
                imgf1_imgf2_overlap_nclt(self.rain_set_imgf1_imgf2_overlap, shuffle=True)

            
            print("=======================================================================\n\n\n")

            print("training with seq: ", np.unique(np.array(self.train_dir1)))
            print("total pairs: ", len(self.train_imgf1))
            print("\n\n\n=======================================================================")

            loss_each_epoch = 0
            used_num = 0

            used_list_f1 = []
            used_list_dir1 = []

            for j in range(len(self.train_imgf1)):
                """
                    check whether the query is used to train before (continue_flag==True/False).
                    TODO: More efficient method
                """
                f1_index = self.train_imgf1[j]
                dir1_index = self.train_dir1[j]
                continue_flag = False
                for iddd in range(len(used_list_f1)):
                    if f1_index == used_list_f1[iddd] and dir1_index == used_list_dir1[iddd]:
                        continue_flag = True
                else:
                    used_list_f1.append(f1_index)
                    used_list_dir1.append(dir1_index)

                if continue_flag:
                    continue

                current_batch = read_one_need_from_seq_nclt(self.data_root_folder, f1_index, dir1_index)   


                sample_batch, sample_truth, pos_num, neg_num = read_one_batch_pos_neg_nclt \
                    (f1_index, dir1_index, self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.train_overlap,
                     self.overlap_thresh, self.data_root_folder)


                """
                    the balance of positive samples and negative samples.
                """
                use_pos_num = 6
                use_neg_num = 6
                if pos_num >= use_pos_num and neg_num >= use_neg_num:
                    sample_batch = torch.cat(
                        (sample_batch[0:use_pos_num, :, :, :], sample_batch[pos_num:pos_num + use_neg_num, :, :, :]),
                        dim=0)
                    sample_truth = torch.cat(
                        (sample_truth[0:use_pos_num, :], sample_truth[pos_num:pos_num + use_neg_num, :]), dim=0)
                    pos_num = use_pos_num
                    neg_num = use_neg_num
                elif pos_num >= use_pos_num:
                    sample_batch = torch.cat((sample_batch[0:use_pos_num, :, :, :], sample_batch[pos_num:, :, :, :]),
                                             dim=0)
                    sample_truth = torch.cat((sample_truth[0:use_pos_num, :], sample_truth[pos_num:, :]), dim=0)
                    pos_num = use_pos_num
                elif neg_num >= use_neg_num:
                    sample_batch = sample_batch[0:pos_num + use_neg_num, :, :, :]
                    sample_truth = sample_truth[0:pos_num + use_neg_num, :]
                    neg_num = use_neg_num

                if neg_num == 0:
                    continue


                current_batch = current_batch.to(self.device)
                sample_batch = sample_batch.to(self.device)
                sample_truth = sample_truth.to(self.device)   

                input_batch = torch.cat((current_batch, sample_batch), dim=0)   

                
                divd =450
                input_batch_trans1 = input_batch[..., :divd]
                input_batch_trans2 = input_batch[..., divd:]
                input_batch_trans = torch.cat([input_batch_trans2,input_batch_trans1],dim=-1)
                input_batch_trans.requires_grad_(True)
                

                input_batch.requires_grad_(True)
                self.amodel.train()
                self.optimizer.zero_grad()


                global_des_add = self.amodel(input_batch, input_batch_trans)   #[13,1024]

                o1, o2, o3 = torch.split(global_des_add, [1, pos_num, neg_num], dim=0)

                MARGIN_1 = 0.5
                """
                    triplet_loss: Lazy for pos
                """
                loss = PNV_loss.triplet_loss(o1, o2, o3, MARGIN_1, lazy=False)
                loss.backward()
                self.optimizer.step()

                current_time = datetime.now()
                formatted_time = current_time.strftime("[%Y-%m-%d %H:%M:%S]")

                if used_num % 1000 == 0:
                    print(formatted_time, str(used_num), loss)

                if torch.isnan(loss):
                    print("Something error ...")
                    print(pos_num)
                    print(neg_num)

                loss_each_epoch = loss_each_epoch + loss.item()
                used_num = used_num + 1

            print("epoch {} loss {}".format(i, loss_each_epoch / used_num))
            print("saving weights ...")
            self.scheduler.step()

            """save trained weights and optimizer states"""
            self.save_name = "./checkpoints/trained_RMT_PR" + str(i) + ".pth.tar"


            torch.save({
                'epoch': i,
                'state_dict': self.amodel.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
                self.save_name)

            print("Model Saved As " + 'trained_RMT_PR' + str(i) + '.pth.tar')

            writer1.add_scalar("loss", loss_each_epoch / used_num, global_step=i)




if __name__ == '__main__':

    config_filename = '../config/config.yml'
    
    config = yaml.safe_load(open(config_filename))
    data_root_folder = config["data_root"]["data_root_folder"]
    training_seqs = config["training_config"]["training_seqs"]

    traindata_npzfiles = config["data_root"]["triplets_for_training"]

    train_handler = trainHandler(height=32, width=900, channels=1, norm_layer=None, lr=0.000005,
                                 data_root_folder=data_root_folder, train_set=traindata_npzfiles,
                                 training_seqs=training_seqs)

    train_handler.train()
