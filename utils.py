import time
import os
import logging
import random
import numpy as np
import torch

def create_logger(args):
    dataset_name = args.dataset
    normal_class = args.normal_class
    neighbor = args.kneighbor
    omega = args.regular
    log_dir = os.path.join(os.path.abspath('.'), "logs")
    log_dir = os.path.join(log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}_class_{}_{}_{}_{}.log".format(dataset_name,normal_class, time_str, neighbor, omega)
    log_file = os.path.join(log_dir, log_name)
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)
    return logger, log_file

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def freeze_parameters(model, train_fc = False):
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return