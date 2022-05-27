# Adapted from github.com/jeonsworld/ViT-pytorch/blob/main/train.py

import logging
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from collections import OrderedDict

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist

import sys
sys.path.insert(0, '../')
from models.resnet_imagenet_2d import resnet18_2d_imagenet, resnet50_2d_imagenet
from models.resnet_imagenet_3d import resnet18_3d_imagenet, resnet50_3d_imagenet
from models.resnet_medicalnet import medicalnet_resnet18
from models.resnet import resnet10, resnet18, resnet34, resnet50
from dataloader import get_loader_kidney
from utils.misc import count_parameters, set_seed

logger = logging.getLogger(__name__)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def setup(args):
    # Prepare model
    num_classes = 2 
    if args.architecture[:6]=='resnet':
        logger.info('RESNET')
        if args.architecture=='resnet18':
            dim_in = 512
            model = resnet18(in_channels=1, sample_size=args.img_size, num_classes=num_classes)
        elif args.architecture=='resnet50':
            dim_in = 2048
            model = resnet50(in_channels=1, sample_size=args.img_size, num_classes=num_classes)
        if args.features_head == 'linear':
            model.fc = nn.Sequential(nn.Linear(dim_in, args.feat_dim), nn.Dropout(args.dropout))
        elif args.features_head == 'mlp':
            model.fc = nn.Sequential(nn.Linear(dim_in, dim_in), nn.Dropout(args.dropout), nn.ReLU(inplace=True), nn.Linear(dim_in, args.feat_dim), nn.Dropout(args.dropout))
        if args.pretrained_features_dir is not None:
            checkpoint = torch.load(args.pretrained_features_dir)
            model.load_state_dict(checkpoint['state_dict'])
        model.fc = Identity()
    elif args.architecture.split('_')[0]=='kinetics':
        logger.info('KINETICCS RESNET')
        dim_in = 512 
        model = generate_model(model_depth=18, n_classes=700)
        if args.pretrained_features_dir is not None:
            checkpoint = torch.load(args.pretrained_features_dir)
            model.load_state_dict(checkpoint['state_dict'])
        model.fc = Identity()
    elif args.architecture.split('_')[0]=='medicalnet':
        logger.info('MEDICALNET RESNET')
        model = medicalnet_resnet18(sample_input_D=14,
                 sample_input_H=28,
                 sample_input_W=28,
                 num_seg_classes=2, 
                 shortcut_type='A').to(args.device)
        model.conv_seg = Identity()
        if args.pretrained_features_dir is not None:
            checkpoint = torch.load(args.pretrained_features_dir)
            new_cp = {}
            for l in checkpoint['state_dict'].keys():
                new_cp[l[7:]] = checkpoint['state_dict'][l]
            model.load_state_dict(new_cp)
            model.conv_seg = nn.AvgPool3d([12, 18, 24], stride=1)
    elif args.architecture.split('_')[0]=='imagenet':
        logger.info('IMAGENET RESNET')
        normalize_mean = np.array([0.485, 0.456, 0.406])
        normalize_std = np.array([0.229, 0.224, 0.225])
        dim_in = 512
        model = resnet18_3d_imagenet()
        if args.pretrained_dir is not None:
            model_2d = resnet18_2d_imagenet()
            model_2d.load_state_dict(torch.load(args.pretrained_dir)['state_dict'])
            new_state_dict = model.state_dict().copy()
            for il, layer in enumerate(model.state_dict().keys()):
                 if model.state_dict()[layer].shape == model_2d.state_dict()[layer].shape:
                     new_state_dict[layer] = model_2d.state_dict()[layer]
                 elif len(model.state_dict()[layer].shape) == len(model_2d.state_dict()[layer].shape)+1:
                     duplicated_weights = torch.unsqueeze(model_2d.state_dict()[layer], dim=-1).repeat_interleave(repeats=model_2d.state_dict()[layer].shape[-1], dim=-1)
                     new_state_dict[layer] = duplicated_weights
            model.load_state_dict(new_state_dict)
            if args.features_head == 'linear':
                model.fc = nn.Sequential(nn.Linear(dim_in, num_classes), nn.Dropout(args.dropout))
            elif args.features_head == 'mlp':
                model.fc = nn.Sequential(nn.Linear(dim_in, args.feat_dim), nn.Dropout(args.dropout), nn.ReLU(inplace=True), nn.Linear(args.feat_dim, num_classes), nn.Dropout(args.dropout))
            model.fc = Identity()
            channel_embedding = nn.Conv3d(1,3,1, bias=True)
            weight = torch.tensor((1/normalize_std)[...,np.newaxis,np.newaxis,np.newaxis,np.newaxis]).float()
            bias = torch.tensor(-normalize_mean/normalize_std).float()
            channel_embedding.weight = torch.nn.Parameter(weight)
            channel_embedding.bias = torch.nn.Parameter(bias)
            model = nn.Sequential(channel_embedding,  model)
        
        if args.pretrained_features_dir is not None:
            model.fc = nn.Sequential(nn.Linear(dim_in, args.feat_dim), nn.Dropout(args.dropout), nn.ReLU(inplace=True), nn.Linear(args.feat_dim, args.feat_dim), nn.Dropout(args.dropout))
            channel_embedding = nn.Conv3d(1,3,1, bias=True)
            model = nn.Sequential(OrderedDict([('channel_embedding', channel_embedding), ('model', model)]))
            checkpoint = torch.load(args.pretrained_features_dir)
            model.load_state_dict(checkpoint['state_dict']) 
            model.model.fc = Identity()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model) 
    model.to(args.device)
    num_params = count_parameters(model)    
   
    return args, model


def train(args, model):
    """ Train the model """
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    # Prepare dataset
    train_loader = get_loader_kidney(args)

    args.num_steps = args.num_epochs * len(train_loader)
    t_total = args.num_steps

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total epochs = %d", args.num_epochs)    
    model.zero_grad()
    set_seed(args)  
    global_step, epoch_step = 0, 0
    features = None
    list_patients = []
    while True:
        epoch_step += 1
        model.eval()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            x = batch['mri']['data'].permute(0, 1, 4, 3, 2)
            x = x.to(args.device)
            if args.architecture.split('_')[0]=='kinetics':
                x = x.repeat(1,3,1,1,1)
            feats = np.squeeze(model(x).detach().cpu().numpy())
            patient_id = np.squeeze(batch['patient_id'])
            list_patients.append(patient_id)
            feats =np.expand_dims(feats, axis=0)
            if features is None:
                features = feats
            else:
                features = np.concatenate([features, feats])
            global_step += 1
            if global_step % t_total == 0:
                break

        if global_step % t_total == 0:
            break
    df_feats = pd.DataFrame(features)
    df_patients = pd.DataFrame(list_patients, columns=['patient'])
    df = pd.concat([df_patients, df_feats], axis=1)
    df.to_csv('./features/features_{}.csv'.format(args.description))
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--exams", default=None, type=str,
                        help="Follow-up exam")
    parser.add_argument('--architecture', default='resnet10', type=str,
                        help='architecture of the model') 
    parser.add_argument('--features_head', choices=['mlp', 'linear'], default='mlp',
                    help='features head architecture')
    parser.add_argument('--feat_dim', default=256, type=int,
                    help='hidden space mlp dimension')

    parser.add_argument("--pretrained_dir", type=str, default=None,
                        help="Where to search for pretrained models.")
    parser.add_argument("--pretrained_features_dir", type=str, default=None,
                        help="Where to search for pretrained features models.")

    parser.add_argument("--img_size", default=[224], nargs='+', type=int,
                        help="Resolution size")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Total batch size for training.")

    parser.add_argument("--num_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--augmentation', default=0, type=int,
                    help='data augmentation transforms')
    parser.add_argument('--dropout', default=0., type=float,
                    help='head dropout')

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--description', type=str,
                        help="Short run description (save features filename)")  
    args = parser.parse_args()
 
    # Setup CUDA, GPU & distributed training
    if torch.cuda.is_available():
        device = "cuda"
        args.n_gpu = torch.cuda.device_count()
        args.device = device
    else:
        device = "cpu"

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO )
    logger.warning("Device: %s, n_gpu: %s" % (args.device, args.n_gpu))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
