import sys
print(sys.path)

import argparse
import math
import os


import torch
import torch.nn as nn
import torch.nn.functional as F

print("torch.__version__ =", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
print('torch.cuda.get_device_name =', torch.cuda.get_device_name)

import numpy as np
import time
import shutil

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.models import resnet50

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE

import math
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA

from SimSiam_8_dataloader import SimSiamDataloader
import torchvision.models as models
import argparse
import torch.backends.cudnn as cudnn

import simsiam.Model
import simsiam.Plot
from simsiam.Plot import scatter
import gc

gc.collect()
torch.cuda.empty_cache()

writer = SummaryWriter("C:\\Users\\KETI\\Triplet2\\summary2")
tsne = TSNE(random_state=0)
device = torch.device('cuda:0')

batch_size = 32

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

def main():
    args = parser.parse_args()

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))    
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    return cur_lr

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):           
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

args = parser.parse_args()

dataset = SimSiamDataloader(r"D:\ShapeNet_simsiam20")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
class_to_idx = dataset.class_to_idx

ratio = 0.8
train_size = int(ratio * len(dataset))
test_size = len(dataset) - train_size
print(f'total: {len(dataset)}\ntrain_size: {train_size}\ntest_size: {test_size}')
train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0
                         )

test_loader = DataLoader(test_data,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0
                         )

resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(resnet50.fc.in_features, 2048)
model = simsiam.Model.SimSiam(resnet50, 2048)
model = model.to(device)

init_lr = args.lr * args.batch_size / 256

criterion = nn.CosineSimilarity(dim=1).to(device)
optim_params = model.parameters()
optimizer = torch.optim.SGD(optim_params, init_lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
cudnn.benchmark = True

tsne_save_path = r"C:\Users\KETI\Triplet2\plot\simsiam\simsiam_class20_0803"

for epoch in range(200):
    running_loss = 0.0

    train_features = []
    train_labels = []

    adjust_learning_rate(optimizer, init_lr, epoch, args)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
    len(train_loader),
    [batch_time, data_time, losses],
    prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        labels = []

        img1, img2, img3, img4, img5, img6, img7, img8, labels = data

        img1 = img1.to(device)
        img2 = img2.to(device)
        img3 = img3.to(device)
        img4 = img4.to(device)
        img5 = img5.to(device)
        img6 = img6.to(device)
        img7 = img7.to(device)
        img8 = img8.to(device)
        #print(f" - img1 shape: {img1.shape}")

        p1, p2, z1, z2 = model(x1=img1, x2=img2, x3=img3, x4=img4, y1=img5, y2=img6, y3=img7, y4=img8)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), img1.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i%2 == 0:
            print(f"At epoches = {epoch}, i = {i}, loss = {loss:.5f}"
                  , end='\r')
            
            writer.add_scalar('training loss',
                              loss, i)
                            #epoch * len(train_loader) + i)
            
        batch_time.update(time.time() - end)
        end = time.time()

        train_features.append(p1.detach().cpu().numpy())
        train_labels.append(labels.detach().cpu().numpy())

    # epoch loss output
    train_epoch_loss = running_loss / len(train_loader)
    print(" "*100)
    print(f"At epoches = {epoch}, epoch_loss = {train_epoch_loss}")

    effective_lr = adjust_learning_rate(optimizer, init_lr, epoch, args)
    writer.add_scalar("learning rate", effective_lr, epoch)

    train_features_np = np.concatenate(train_features, axis=0)
    train_labels_np = np.concatenate(train_labels, axis=0)

    train_features_embedded = tsne.fit_transform(train_features_np)


    fig, ax = plt.subplots()
    scatter(train_features_embedded, train_labels_np)

    if epoch % 1 == 0:
        plt.savefig(os.path.join(tsne_save_path, f"tsne_epoch{epoch}.png"), dpi=300, bbox_inches='tight')
    
    if epoch % 10 == 0:
        model_path = os.path.join(tsne_save_path, f"simsiam_model_epoch{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Trained model saved at epoch {epoch}: {model_path}")
    
test_features = []
test_labels = []

model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = []
        labels = []

        img1, img2, img3, img4, img5, img6, img7, img8, labels = data
        img1 = img1.to(device)
        img2 = img2.to(device)
        img3 = img3.to(device)
        img4 = img4.to(device)
        img5 = img5.to(device)
        img6 = img6.to(device)
        img7 = img7.to(device)
        img8 = img8.to(device)

        p1, p2, z1, z2 = model(x1=img1, x2=img2, x3=img3, x4=img4, y1=img5, y2=img6, y3=img7, y4=img8)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), img1.size(0))

        running_loss += loss.item()
        if i%2 == 0:
            print(f"Evaluating at epoches = {epoch}, i = {i}, loss = {loss:.5f}", end='\r')
                
            writer.add_scalar('test loss',
                            running_loss / 1000,
                            epoch * len(test_loader) + i)
        test_features.append(p1.detach().cpu().numpy())
        test_labels.append(labels.detach().cpu().numpy())

    test_epoch_loss = running_loss / len(train_loader)
    print(" "*100)
    print(f"At epoches = {epoch}, epoch_train_loss = {train_epoch_loss}, \tepoch_test_loss = {test_epoch_loss}")

    test_features_np = np.concatenate(test_features, axis=0)
    test_labels_np = np.concatenate(test_labels, axis=0)
        
    test_features_embedded = tsne.fit_transform(test_features_np)

    fig, ax = plt.subplots()
    scatter(test_features_embedded, test_labels_np)
    writer.add_figure('TSNE plot', fig, global_step=epoch)
    plt.savefig(os.path.join(tsne_save_path, f"test_tsne.png"), dpi=300, bbox_inches='tight')

    print("Finished Training")
    writer.close()