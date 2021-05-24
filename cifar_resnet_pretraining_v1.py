'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import os
import argparse
from torch.utils.data import TensorDataset
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, random_split

from experiments.models import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda'
best_acc = 90  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
def get_data_utils(batch_size,chunks,num_chunk):
    scale=40
    dataset_fun = CIFAR10
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

print('==> Preparing data..')

'''transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])'''

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=Compose([ToTensor()]))
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def get_model():
    # Model
    print('==> Building model Resnet18')
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    print('==> Resuming from checkpoint..')
    #assert os.path.isdir('experiments/checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('/home/xqq/5-7/5-17/experiments/checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    #best_acc = checkpoint['acc']
    #start_epoch = checkpoint['epoch']
    return net


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    '''acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc'''

'''net=get_model()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()
''''''
for epoch in range(start_epoch, start_epoch+2):
    clean_acc = get_clean_acc(net, testloader, device)
    #test(epoch)
    scheduler.step()

testloader, start_ind, end_ind = get_data_utils(250,10, 
                                                    1)
# Clean acc
clean_acc = get_clean_acc(net, testloader, device)'''
