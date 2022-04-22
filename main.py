from network.attention_module import Attention
from network.resnet import get_resnet_model
from dataset.main import load_dataset
from utils import create_logger,seed,freeze_model
from bank import get_bank, knn_score
from eval import detection_test
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import os
import argparse
import time
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings("ignore")
seed(0)

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser(description= 'Settings in SAP2')
parser.add_argument('--data_path', default = './data', type = str)
parser.add_argument('--normal_class', default =0, type = int)
parser.add_argument('--epochs', default = 10, type = int)
parser.add_argument('--dataset', default = 'cifar100', type = str)
parser.add_argument('--batchsize', default = 64, type = int)
parser.add_argument('--kneighbor', default = 32, type = int)
parser.add_argument('--lr', default = 5e-4, type = float)
parser.add_argument('--regular', default = 2.0, type = float)
args = parser.parse_args()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

logger, log_file = create_logger(args)
logger.info('dataset {} batchsize {} lr {}'.format(args.dataset, args.batchsize, args.lr))

# model
ENCODER = get_resnet_model(resnet_type= 152).to(device)
freeze_model(ENCODER)
ATTENTION = Attention(2048, 2048).to(device)
ENCODER.eval()
# optim
param_list = [
    {'params':ATTENTION.parameters(), 'lr':args.lr}
]
optimizer = optim.Adam(param_list)
for param_group in optimizer.param_groups:
    logger.info('learning rate {}'.format(param_group['lr']))
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # cifar 
# dataset

dataset = load_dataset(args.dataset, args.data_path, args.normal_class)
train_loader,testloader = dataset.loaders(batch_size=args.batchsize ,shuffle_train=True, shuffle_test=False)

# bank
z_list, z_list_norm = get_bank(ENCODER, train_loader, logger, device)

# loss
similarity_loss = torch.nn.CosineSimilarity()
mse_loss = torch.nn.MSELoss()

for epoch in range(args.epochs):
    train_loss = 0
    n_batch = 0
    epoch_start_time = time.time()
    train_score = []
    ATTENTION.train()

    for x, label in tqdm(train_loader):
        x = x.to(device)
        optimizer.zero_grad()
        a,z,f_map = ENCODER(x) # featuremap
        Distance,Index = knn_score(z_list_norm, z, z_list.shape[0])
        new_z_list = []
        for i in range(z.shape[0]):
            newz = z_list[Index[i][1:args.kneighbor + 1]].unsqueeze(0)
            new_z_list.append(newz)
        new_z_list = torch.cat(new_z_list, dim = 0)
        z_aug, z_aug_new, z, z_new= ATTENTION(new_z_list, z)
        loss_COS2 = torch.mean(1 - similarity_loss(z_aug_new, z_new))
        regular = mse_loss(z, z_new) + torch.mean(1 - similarity_loss(z, z_new))
        loss = loss_COS2 + args.regular*regular
        loss.backward(retain_graph=True)
        train_loss += loss.item()
        n_batch += 1
        optimizer.step()

    epoch_train_time = time.time() - epoch_start_time
    scheduler.step()
    logger.info('epoch {}  loss {} Time: {:.3f}\t'.format(epoch, train_loss/n_batch, epoch_train_time))
detection_test(args, ENCODER, ATTENTION, testloader, z_list, z_list_norm, device)
