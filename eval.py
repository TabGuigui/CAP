import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
from bank import knn_score
from tqdm import tqdm
import logging

def detection_test(args, encoder, attention, test_dataloader, z_list, z_list_norm, device):
    logger = logging.getLogger()
    encoder.eval()
    attention.eval()
    similarity_loss = nn.CosineSimilarity()
    score_list = []
    normal_list = []
    abnormal_list = []
    train_num = z_list.shape[0]
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            a,z,f_map = encoder(inputs)
            Distance,Index = knn_score(z_list_norm, z, train_num) # Index batch * neighbor
            new_z_list = []
            for i in range(z.shape[0]):
                newz = z_list[Index[i][:args.kneighbor]].unsqueeze(0)
                new_z_list.append(newz)
            
            new_z_list = torch.cat(new_z_list, dim = 0)
            z_aug, z_aug2, z, z_2 = attention(new_z_list,z)
            loss_cos = (1 - similarity_loss(z_aug2, z_2))  
            score = loss_cos

            score_list += list(zip(labels.cpu().data.numpy().tolist(), score.cpu().data.numpy().tolist()))
            normal_list.append(score[np.where(labels == 0)])
            abnormal_list.append(score[np.where(labels == 1)])     

    normal_list = torch.cat(normal_list, dim = 0)
    abnormal_list = torch.cat(abnormal_list, dim = 0)
    logger.info('normal **** {}'.format(torch.mean(normal_list)))
    logger.info('abnormal ****{}'.format(torch.mean(abnormal_list)))
    labels, scores = zip(*score_list)
    labels = np.array(labels)
    scores = np.array(scores)
    # print(labels, scores)
    test_auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fpr95 = fpr[(np.abs(tpr - 0.95)).argmin()]
    logger.info('AUC {} '.format(test_auc))
    