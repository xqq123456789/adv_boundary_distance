import os
import random
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from scipy import ndimage
from autoattack import AutoAttack

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import TensorDataset
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, random_split
import torchvision.models as models 

def get_data_utils(dataset_name, batch_size, chunks, num_chunk):
    scale=100
    #scale=1
    if dataset_name == 'imagenet':
        from torchvision import transforms
        path = '/local/reference/CV/ILSVR/classification-localization/data/jpeg/val'
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        dataset = ImageFolder(path, transform)
    else:
        dataset_fun = CIFAR10 if dataset_name == 'cifar10' else CIFAR100
        dataset = dataset_fun(root='./data', train=False, download=True,
                            transform=Compose([ToTensor()]))
    tot_instances = len(dataset)
    #print('lols', tot_instances)
    assert 1 <= num_chunk <= chunks
    assert tot_instances % chunks == 0
    #print(chunks,num_chunk)
    # inds of current chunk
    inds = np.linspace(0, tot_instances, chunks+1, dtype=int) #等差数列
    #print(inds)
    start_ind, end_ind = inds[num_chunk-1], inds[num_chunk]
    # extract data and put in new dataset
    data = [dataset[i] for i in range(start_ind, end_ind)]
    imgs = torch.cat([x.unsqueeze(0) for (x, y) in data], 0)[:int(tot_instances/scale)]
    #print(imgs.size())
    labels = torch.cat([torch.tensor(y).unsqueeze(0) for (x, y) in data], 0)[:int(tot_instances/scale)]
    testset = TensorDataset(imgs, labels)
    #print(imgs.size())

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True, drop_last=False)
    #print(start_ind, end_ind)
    return testloader, int(start_ind/scale), int(end_ind/scale)


def get_clean_acc(model, testloader, device):
    model.eval()
    n, total_acc = 0, 0
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            #print(X,y)
            #print(X[0][0])
            output = model(X)
            #print(output)
            total_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    
    acc = 100. * total_acc / n
    print(f'Clean accuracy: {acc:.4f}')
    return acc


def get_adversary(model, cheap, seed, eps):
    model.eval()
    adversary = AutoAttack(model.forward, norm='Linf', eps=eps, verbose=False)
    #adversary = AutoAttack(model.forward, norm='L2', eps=eps, verbose=False)
    adversary.seed = seed
    return adversary

def compute_advs(model, testloader, device, batch_size, cheap, seed, eps):
    model.eval()
    adversary = get_adversary(model, cheap, seed, eps)
    #print('advs_lalal')
    imgs = torch.cat([x for (x, y) in testloader], 0)
    labs = torch.cat([y for (x, y) in testloader], 0)
    adversary.attacks_to_run = ['apgd-ce']
    #print("lala",imgs.shape)
    advs = adversary.run_standard_evaluation_individual(imgs, labs, 
                                                        bs=batch_size)#这一步不断去找model,并且利用梯度进行攻击
    #print("adv:",advs[0][0])
    #print((imgs-advs['apgd-ce']).size())
    return advs, labs

def compute_adv_accs(model, advs, labels, device, batch_size):
    accs = {}
    all_preds = []
    for attack_name, curr_advs in advs.items():
        dataset = TensorDataset(curr_advs, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=1, pin_memory=True, drop_last=False)
        total_corr = 0
        curr_preds = []
        with torch.no_grad():
            for img, lab in dataloader:
                img, lab = img.to(device), lab.to(device)
                #print(img, lab)
                #print(img[0][0])
                output = model(img)
                pred = output.max(1)[1]
                curr_preds.append(pred)
                total_corr += (pred == lab).sum().item()

        curr_preds = torch.cat(curr_preds)
        all_preds.append(curr_preds)
            
        curr_acc = 100. * total_corr / labels.size(0)
        accs.update({ attack_name : curr_acc })
    
    # compute worst case for each image
    all_preds = torch.cat([x.unsqueeze(0) for x in all_preds])
    temp_labels = labels.unsqueeze(0).expand(len(advs), -1).to(device)
    where_all_correct = torch.prod(all_preds==temp_labels, dim=0) # logical AND
    worst_acc = 100. * where_all_correct.sum().item() / labels.size(0)
    accs.update({ 'rob acc' : worst_acc })
    print('adv_acc:',worst_acc)

    return accs

def get_model():
    from experiments.cifar_resnet_pretraining_v1 import get_model
    model = get_model()
    model.to("cuda")
    return model

