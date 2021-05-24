import sys
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
#import cv2
import numpy as np
from torch.autograd import Variable
import copy
import time
#from pyheatmap.heatmap import HeatMap
import pandas as pd
import seaborn as sns
from PIL import Image
import argparse
import datetime
#from architectures import ARCHITECTURES, get_architecture
#from attacks import Attacker, PGD_L2, DDN
#from datasets import get_dataset, DATASETS, get_num_classes,\
                     #MultiDatasetsDataLoader, TiTop50KDataset
import numpy as np
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
#from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code, requires_grad_
import random
import argparse
import os.path as osp
import torch.backends.cudnn as cudnn
from utils5_24 import (get_model, print_to_log, eval_chunk,
                         eval_files,get_data_utils,get_clean_acc,compute_advs,compute_adv_accs)
from distance5_24v1 import (High_quality_boundaries,binary_search,seed_point,seed_quality,boundary_quality,compute_adv_distance,compute_ori_distance)

def set_seed(device, seed=111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)              

def main(args):
    # Setup
    seed=0
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(DEVICE, seed)

    # Model
    model = get_model()
    model.eval()#don't change the parameter
    #test dataset
    testloader,start_ind, end_ind = get_data_utils(args.dataset, args.batch_size, args.chunks, 
                                                    1) #num_chunk
    
    for seed_num in range(10):
        for m in range(1,3):
            #seed_data
            seed_data,seed_label=seed_point(testloader,model,seed_num,DEVICE,m)
            #pre_data
            #pre_data,pre_label=seed_point(testloader,model,2,DEVICE)
    
            #clean_acc
            get_clean_acc(model, testloader, DEVICE)

            #adv
            advs,labels=compute_advs(model, testloader, DEVICE, args.batch_size, args.cheap, seed, args.eps)
            #adv_acc
            compute_adv_accs(model, advs, labels, DEVICE, args.batch_size)
            boundary=High_quality_boundaries(model,testloader,seed_data,seed_label,DEVICE)

            #txt
            full_path = '/home/xqq/5-7/5-17/result/resnet_lin_eps_0.031/' + '5_24_sample100_m'+str(m)+'_seed_num'+str(seed_num)+'.txt'
            print(full_path)
            with open(full_path,'a') as f: 
                for i in range(len(boundary)):
                    [target_label,w,b]=boundary[i]
                    target_label,boundary_accuracy,cnn_accuracy=boundary_quality(testloader,w,b,seed_label,target_label,model,DEVICE)
                    print(boundary_accuracy,cnn_accuracy)
                    if boundary_accuracy>0.75:
                        adv_distance=compute_adv_distance(model, advs, labels, DEVICE, args.batch_size, seed_label,target_label,w,b)
                        ori_distance=compute_ori_distance(model, testloader, DEVICE, args.batch_size, seed_label,target_label,w,b)
                        f.write('seed_label:'+str(seed_label))
                        f.write('\t')
                        f.write('target_label:'+str(target_label))
                        f.write('\n')
                        f.write('ori_distance')
                        f.write('\n')
                        np.savetxt(f,ori_distance)
                        f.write('adv_distance')
                        f.write('\n')
                        np.savetxt(f,adv_distance)
                        print(target_label)

    print("yes model")

if __name__ == "__main__":
    from opts import parse_settings
    args = parse_settings()
    main(args)
