import sys
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from torch.autograd import Variable
import copy
import time
import pandas as pd
import seaborn as sns
from PIL import Image
import argparse
import datetime
import numpy as np
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, random_split

#Select a fixed-label sample in train_data as seed_data, and remove the inaccurate prediction
def seed_point(testloader,model,seed_label,device,M):
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            m=0
            for i in enumerate(y):
                if int(i[1])==seed_label:
                    x_s=torch.unsqueeze(x[i[0]], dim=0).type(torch.FloatTensor).to(device)
                    if model(x_s).max(1)[1]==seed_label:
                        m+=1
                        #print(seed_label)
                        if m==M:
                            return x_s,seed_label
    return False

#High-quality boundaries of one seed
def High_quality_boundaries(model,testloader,seed_data,seed_label,device):
    model.eval()
    boundary=[]
    NUM=0 #number of high-quality boundaries
    with torch.enable_grad():#because usually people disables gradients in evaluations!
        for x, y in testloader:
            x, y = x.to(device), y.to(device)#data_num/batch_size
            for i in enumerate(y):
                if int(i[1])!=seed_label:
                    x_s=torch.unsqueeze(x[i[0]], dim=0).type(torch.FloatTensor).to(device)
                    if model(x_s).max(1)[1]==int(i[1]):#Remove inaccurate predictions
                        pre_data=x_s
                        pre_label=int(i[1])
                        s_i,num,good_boundary=binary_search(seed_data,pre_data,seed_label,pre_label,model,0.9,0.001,device)
                        NUM+=good_boundary
                        if num<1000 and good_boundary==1:
                            out=model(s_i)
                            #print(int(i[1]))
                            boundary=one_boundary(out,s_i,boundary,int(i[1]),device)
    #print(NUM)
    return boundary

#binary_rearch,(|pos_num-neg_num|<e) and (pred_y==pos_label or neg_label)
def binary_search(x_pos,x_neg,pos_label,neg_label,model,r,e,device):
    model.eval()
    num=0
    good_boundary=0
    fs_i=model(x_neg)
    while True:
        if num>1000:
            break;
        s_i=(r*x_pos+(1-r)*x_neg).to(device)
        fs_i=model(s_i)
        pos_num=fs_i[0][pos_label]
        neg_num=fs_i[0][neg_label]
        if abs(pos_num-neg_num)<e:
            break;
        if pos_num>neg_num:
            x_pos=s_i
            num+=1
        else:
            x_neg=s_i
            num+=1
    s_i=Variable(s_i,requires_grad=True)#Need to declare gradient required
    pred_y= torch.max(fs_i, 1)[1].data.cpu().numpy()
    if pred_y[0]==pos_label or pred_y[0]==neg_label:
        good_boundary=1
        #print("last_true:",pred_y[0],"pos:",pos_label,"neg:",neg_label,"num:",num)
        #print(fs_i)
    return s_i,num,good_boundary

#find one boundary——H(label of s_i,w,b)
def one_boundary(out,s_i,boundary,label,device):
    w=gradient(out,s_i,device).to(device)
    multi=multi_x_w(s_i,w).to(device)
    b=out-multi
    H=[label,w,b]
    boundary.append(H)
    return boundary	

#Find the gradient of the input relative to the output
def gradient(out,s_i,device):	
    for i in range(0,10):
        x=torch.tensor([[0,0,0,0,0,0,0,0,0,0]],dtype=torch.float).to(device)
        x[0][i]=1
        out.backward(gradient=x,retain_graph=True)
        temp=copy.deepcopy(s_i.grad)
        #print(temp.size())
        s_i.grad.data.zero_()
        if i>0:
            temp=torch.cat((temp1,temp),0)
        temp1=temp
    return temp

#Find the product of the input x and the gradient w
def multi_x_w(x,w):
    multi=torch.tensor([[0,0,0,0,0,0,0,0,0,0]],dtype=torch.float)
    for m in range(0,x.size()[0]):
        for i in range(0,10):
            sum_num=0
            for j in range(0,3):
                for k in range(0,32):
                    for g in range(0,32):
                        sum_num+=x[m][j][k][g]*w[i][j][k][g]
            multi[0][i]=sum_num
        if m==0:
            multi_last=multi
        else:
            multi_last=torch.cat((multi_last,multi),0)
    return multi_last

#Determine whether the seed is correctly classified by the boundary
def seed_quality(w,b,x_s_label,x_s,target_label,model,device):
    model.eval()
    #x_s=torch.unsqueeze(x_s, dim=0).type(torch.FloatTensor).to(device)
    mul_r=multi_x_w(x_s,w).to(device)
    result=mul_r+b
    out=model(x_s)
    #print('reuslt:',result,'b:',b,target_label)
    pred_y= torch.max(out, 1)[1].data.cpu().numpy()
    if result[0][x_s_label]>result[0][target_label]:
        return 1
    else:
        return 0

#check the quality of boundary
def boundary_quality(testloader,w,b,seed_label,target_label,model,device):
    model.eval()
    cnn_num=0
    total_num=0
    true_num=0
    with torch.no_grad():#because usually people disables gradients in evaluations!
        for x, y in testloader:
            x, y = x.to(device), y.to(device)#data_num/batch_size
            for i in enumerate(y):
                if int(i[1])==seed_label or int(i[1])==target_label:
                    total_num+=1
                    x_s=torch.unsqueeze(x[i[0]], dim=0).type(torch.FloatTensor).to(device)
                    mul_s=multi_x_w(x_s,w).to(device)
                    result=mul_s+b
                    out=model(x_s)
                    pred_y= torch.max(out, 1)[1].data.cpu().numpy()
                    if (int(i[1])==seed_label and result[0][seed_label]>result[0][target_label]) or (int(i[1])==target_label and result[0][seed_label]<result[0][target_label]):
                        true_num+=1
                    if int(i[1])==pred_y:
                        cnn_num+=1
    return target_label,true_num/total_num,cnn_num/total_num

#Find the distance from the adversarial sample to the boundary
def compute_adv_distance(model, advs, labels, device, batch_size,seed_label,target_label,w,b):
    for attack_name, curr_advs in advs.items():
        dataset = TensorDataset(curr_advs, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=1, pin_memory=True, drop_last=False)
        with torch.no_grad():
            for img, lab in dataloader:                
                img, lab = img.to(device), lab.to(device)
                #print(img, lab)
                #print(img[0])
                svm_dis=svm_distance(w,b,img,device)
                distance=torch.norm(svm_dis,dim=1).detach().cpu().numpy()
                #print(torch.norm(svm_dis[0]).detach().cpu().numpy())
                #print("adv_distance:",distance)
    return distance

#Find the distance from the original sample to the boundary
def compute_ori_distance(model, testloader, device, batch_size, seed_label,target_label,w,b):
    with torch.no_grad():#because usually people disables gradients in evaluations!
        for x, y in testloader:
            img, lab = x.to(device), y.to(device)#data_num/batch_size
            #print('img:',img[0][0])
            svm_dis=svm_distance(w,b,img,device)
            distance=torch.norm(svm_dis,dim=1).detach().cpu().numpy()
            #print("ori_distance:",distance)
    return distance

#Find svm distance
def svm_distance(w,b,data,device):
    #data=torch.unsqueeze(data, dim=1).type(torch.FloatTensor).to(device)
    Denominator= torch.norm(data)
    molecular=abs(multi_x_w(data,w).to(device)+b.to(device))
    #print(molecular/Denominator)
    return molecular/Denominator

