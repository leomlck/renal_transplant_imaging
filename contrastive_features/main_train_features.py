# Adapted from github.com/jeonsworld/ViT-pytorch/blob/main/train.py

import logging
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm

from models.resnet import resnet10, resnet18, resnet34, resnet50
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from dataloader import get_loader_kidney_patient_disc, get_loader_kidney_mdrd_disc, get_loader_kidney_age_donor_disc, get_loader_kidney_fibrose_disc, get_loader_kidney_incomp_disc, get_loader_kidney_citime_disc


from torch.nn.modules.utils import _pair, _triple

import subprocess
from io import BytesIO
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, average_precision_score
#from torchsummary import summary
import wandb
from skimage.transform import resize
import io
from collections import OrderedDict

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

def roc_auc(y_true, y_pred):
        try:
                return roc_auc_score(y_true, y_pred)
        except ValueError:
                return 0

def save_ckp(args, ckp, is_best=False):
    # Save model checkpoint
    model_checkpoint = os.path.join(args.output_dir, args.wandb_id, "%s_checkpoint.bin" % args.name)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
    torch.save(ckp, model_checkpoint)
    if is_best:
        model_checkpoint = os.path.join(args.output_dir, args.wandb_id, "%s_best.bin" % args.name)
        torch.save(ckp, model_checkpoint)
        logger.info("Saved best model checkpoint to [DIR: %s]", args.output_dir)

def load_ckp(args, model, optimizer, scheduler):
    # Load model checkpoint
    checkpoint = torch.load(os.path.join(args.output_dir, args.wandb_id, "%s_checkpoint.bin" % args.name))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['wandb_step'], checkpoint['global_step'], checkpoint['epoch_step'], checkpoint['best_loss'], checkpoint['curriculum_it']

def setup(args):
    # Prepare model
    if args.architecture[:6]=='resnet':
        if args.architecture=='resnet18':
            dim_in = 512
            model = resnet18(in_channels=1, sample_size=args.img_size)
        elif args.architecture=='resnet50':
            dim_in = 2048
            model = resnet50(in_channels=1, sample_size=args.img_size)
        # Replace classification head
        if args.features_head == 'linear':
            model.fc = nn.Sequential(nn.Linear(dim_in, args.feat_dim), nn.Dropout(args.dropout))
        elif args.features_head == 'mlp':
            model.fc = nn.Sequential(nn.Linear(dim_in, dim_in), nn.Dropout(args.dropout), nn.ReLU(inplace=True), nn.Linear(dim_in, args.feat_dim), nn.Dropout(args.dropout))
   if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model) 
    model.to(args.device)
    num_params = count_parameters(model)    
   
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)      
       
def valid(args, model, eval_loader, wandb_step, global_step, epoch_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("\n\n***** Running Validation *****")
    logger.info("  Num steps = %d", len(eval_loader))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    all_preds, all_preds_prob, all_label = [], [], []
    epoch_iterator = tqdm(eval_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)
    sm = torch.nn.Softmax(dim=1)
    cos_sim = torch.nn.CosineSimilarity()
    loss_fct = torch.nn.CosineEmbeddingLoss(args.loss_margin) 
    for step, batch in enumerate(epoch_iterator):
        x1 = batch['mri1']['data'].permute(0, 1, 4, 3, 2).float()
        x2 = batch['mri2']['data'].permute(0, 1, 4, 3, 2).float()
        y = batch['label'].float()
        x1, x2, y = x1.to(args.device), x2.to(args.device), y.to(args.device)
        with torch.no_grad():
            bsz = y.shape[0]
            x = torch.cat([x1, x2], dim=0)
            features = model(x)
            v1, v2 = torch.split(features, [bsz, bsz], dim=0)
            if args.normalize_feat:
                v1 = nn.functional.normalize(v1, p=2, dim=1)
                v2 = nn.functional.normalize(v2, p=2, dim=1)
            eval_loss = loss_fct(v1, v2, y)
            preds_prob = 2*cos_sim(v1, v2).float()-1
            preds = (preds_prob>0.5).float()
            y = (y>0).float()
            eval_losses.update(eval_loss.item())
        
        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            all_preds_prob.append(preds_prob.detach().cpu().numpy()) 
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            all_preds_prob[0] = np.append(
                all_preds_prob[0], preds_prob.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
 
    all_preds, all_preds_prob, all_label = all_preds[0], all_preds_prob[0], all_label[0]
    eval_accuracy = accuracy_score(all_label, all_preds)
    eval_precision = precision(all_label, all_preds)
    eval_recall = recall(all_label, all_preds)
    eval_f1 = f1(all_label, all_preds)
    eval_roc_auc = roc_auc(all_label, all_preds_prob)
    eval_ap = average_precision_score(all_label, all_preds_prob)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % eval_accuracy)
    logger.info("Valid ROC AUC: %2.5f" % eval_roc_auc)

    wandb.log({'validation/loss': eval_losses.avg,
        'validation/accuracy': eval_accuracy,
        'validation/precision': eval_precision,
        'validation/recall': eval_recall,
        'validation/f1': eval_f1,
        'validation/roc_auc': eval_roc_auc,
        'validation/AP': eval_ap,
        'global_step': global_step,
        'epoch_step': epoch_step}, step=wandb.run.step+wandb_step+1)

    return eval_losses.avg

def train(args, model):
    """ Train the model """
    # Prepare dataset
    if args.target=='mdrd':
        train_loader,_ = get_loader_kidney_mdrd_disc(args, curriculum=[args.curriculum[0], args.curriculum[1]])
        _, eval_loader = get_loader_kidney_mdrd_disc(args, curriculum=[args.curriculum[-2], args.curriculum[-1]])
    elif args.target=='patient':    
        train_loader,_ = get_loader_kidney_patient_disc(args, curriculum_step=args.curriculum[0])
        _, eval_loader = get_loader_kidney_patient_disc(args, curriculum_step=args.curriculum[-1])
    elif args.target=='age_donor':
        train_loader,_  = get_loader_kidney_age_donor_disc(args, curriculum=[args.curriculum[0], args.curriculum[1]])
        _, eval_loader = get_loader_kidney_age_donor_disc(args, curriculum=[args.curriculum[-2], args.curriculum[-1]])
    elif args.target=='fibrose':
        train_loader,_  = get_loader_kidney_fibrose_disc(args, curriculum=[args.curriculum[0], args.curriculum[1]])
        _, eval_loader = get_loader_kidney_fibrose_disc(args, curriculum=[args.curriculum[-2], args.curriculum[-1]])
    elif args.target=='incomp_gref':
        train_loader,_  = get_loader_kidney_incomp_disc(args, curriculum=[args.curriculum[0], args.curriculum[1]])
        _, eval_loader = get_loader_kidney_incomp_disc(args, curriculum=[args.curriculum[-2], args.curriculum[-1]])
    elif args.target=='cold_ischem_time':
        train_loader,_  = get_loader_kidney_citime_disc(args, curriculum=[args.curriculum[0], args.curriculum[1]])
        _, eval_loader = get_loader_kidney_citime_disc(args, curriculum=[args.curriculum[-2], args.curriculum[-1]])
 
    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    args.num_steps = args.num_epochs[-1] * len(train_loader)
    args.warmup_steps = args.warmup_epochs * len(train_loader)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total epochs = %d", args.num_epochs[-1])
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info("  Train batch size = %d", args.batch_size)
    logger.info("  Using Cosine Embedding Loss")
    
    model.zero_grad()
    set_seed(args)  
    losses = AverageMeter()
    wandb_step, global_step, epoch_step, best_loss = 0, 0, 0, 1e12
    curriculum_it = 0
    if args.resume:
        model, optimizer, scheduler, wandb_step, global_step, epoch_step, best_loss, curriculum_it = load_ckp(args, model, optimizer, scheduler)
        if args.target=='mdrd':
            train_loader,_ = get_loader_kidney_mdrd_disc(args, curriculum=[args.curriculum[2*curriculum_it], args.curriculum[2*curriculum_it+1]])
            _, eval_loader = get_loader_kidney_mdrd_disc(args, curriculum=[args.curriculum[-2], args.curriculum[-1]])
        elif args.target=='patient':
            train_loader,_ = get_loader_kidney_patient_disc(args, curriculum_step=args.curriculum[curriculum_it])
            _, eval_loader = get_loader_kidney_patient_disc(args, curriculum_step=args.curriculum[-1])
        elif args.target=='age_donor':
            train_loader,_ = get_loader_kidney_age_donor_disc(args, curriculum=[args.curriculum[2*curriculum_it], args.curriculum[2*curriculum_it+1]])
            _, eval_loader = get_loader_kidney_age_donor_disc(args, curriculum=[args.curriculum[-2], args.curriculum[-1]])
        elif args.target=='fibrose':
            train_loader,_ = get_loader_kidney_fibrose_disc(args, curriculum=[args.curriculum[2*curriculum_it], args.curriculum[2*curriculum_it+1]])
            _, eval_loader = get_loader_kidney_fibrose_disc(args, curriculum=[args.curriculum[-2], args.curriculum[-1]])
        elif args.target=='incomp_gref':
            train_loader,_ = get_loader_kidney_incomp_disc(args, curriculum=[args.curriculum[2*curriculum_it], args.curriculum[2*curriculum_it+1]])
            _, eval_loader = get_loader_kidney_incomp_disc(args, curriculum=[args.curriculum[-2], args.curriculum[-1]])
        elif args.target=='cold_ischem_time':
            train_loader,_ = get_loader_kidney_citime_disc(args, curriculum=[args.curriculum[2*curriculum_it], args.curriculum[2*curriculum_it+1]])
            _, eval_loader = get_loader_kidney_citime_disc(args, curriculum=[args.curriculum[-2], args.curriculum[-1]])
    while True:
        t = time.time()
        epoch_step += 1
        if epoch_step > args.num_epochs[curriculum_it]:
            curriculum_it += 1
            if args.target=='mdrd':
                train_loader,_ = get_loader_kidney_mdrd_disc(args, curriculum=[args.curriculum[2*curriculum_it], args.curriculum[2*curriculum_it+1]])
            elif args.target=='patient':
                train_loader,_ = get_loader_kidney_patient_disc(args, curriculum_step=args.curriculum[curriculum_it])
            elif args.target=='age_donor':
                train_loader,_ = get_loader_kidney_age_donor_disc(args, curriculum=[args.curriculum[2*curriculum_it], args.curriculum[2*curriculum_it+1]])
            elif args.target=='fibrose':
                train_loader,_ = get_loader_kidney_fibrose_disc(args, curriculum=[args.curriculum[2*curriculum_it], args.curriculum[2*curriculum_it+1]])
            elif args.target=='incomp_gref':
                train_loader,_ = get_loader_kidney_incomp_disc(args, curriculum=[args.curriculum[2*curriculum_it], args.curriculum[2*curriculum_it+1]])
            elif args.target=='cold_ischem_time':
                train_loader,_ = get_loader_kidney_citime_disc(args, curriculum=[args.curriculum[2*curriculum_it], args.curriculum[2*curriculum_it+1]])
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=False) 
        loss_fct = torch.nn.CosineEmbeddingLoss(args.loss_margin) 
        for step, batch in enumerate(epoch_iterator):
            x1 = batch['mri1']['data'].permute(0, 1, 4, 3, 2).float()
            x2 = batch['mri2']['data'].permute(0, 1, 4, 3, 2).float()
            y = batch['label'].float()
            x1, x2, y = x1.to(args.device), x2.to(args.device), y.to(args.device)
            bsz = y.shape[0]
            x = torch.cat([x1, x2], dim=0)
            features = model(x)
            v1, v2 = torch.split(features, [bsz, bsz], dim=0)
            if args.normalize_feat:
                v1 = nn.functional.normalize(v1, p=2, dim=1)
                v2 = nn.functional.normalize(v2, p=2, dim=1)
            loss = loss_fct(v1, v2, y)
            loss.backward()

            losses.update(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)) 
            # Save model checkpoint every 2 hours -- if epochs are too long.
            if (time.time()-t)/60 > 120:
                 ckp = {'wandb_step': wandb.run.step,
               'global_step': global_step,
               'epoch_step': epoch_step,
               'best_loss': best_loss,
               'curriculum_it': curriculum_it,
               'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict()}
                 save_ckp(args, ckp, is_best=False)
                 t = time.time()

            if global_step % t_total == 0:
                break

        if epoch_step % args.eval_every == 0:
            eval_loss = valid(args, model, eval_loader, wandb_step, global_step, epoch_step)
            if best_loss >= eval_loss:
                ckp = {'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()}
                save_ckp(args, ckp, is_best=True)
                best_loss = eval_loss
            model.train()

        wandb.log({'train/epoch_loss': losses.avg, 'global_step': global_step, 'epoch_step': epoch_step}, step=wandb.run.step+wandb_step+1)
        wandb.log({'train/lr': scheduler.get_last_lr()[0], 'global_step': global_step, 'epoch_step': epoch_step}, step=wandb.run.step+wandb_step+1)
        losses.reset()
        ckp = {'wandb_step': wandb.run.step,
               'global_step': global_step,
               'epoch_step': epoch_step,
               'best_loss': best_loss,
               'curriculum_it': curriculum_it,
               'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict()}
        save_ckp(args, ckp, is_best=False)
        if global_step % t_total == 0:
            break

    logger.info("Best Eval Loss: \t%f" % best_loss)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--target", choices= ['mdrd', 'patient', 'age_donor', 'fibrose', 'incomp_gref', 'cold_ischem_time'], type=str,
                        help="Which upstream task.")
    parser.add_argument("--dataset_size", default=1000, type=int,
                        help="Size of the data set")
    parser.add_argument("--testset_size", default=100, type=int,
                        help="Size of the validation set")
    parser.add_argument("--features_head", choices=["mlp", "linear"], default="mlp",
                        help="Output features head model")
    parser.add_argument("--feat_dim", default=256, type=int,
                        help="Output features space dimension")    
    parser.add_argument('--architecture', metavar='ARCH', default='resnet18', type=str,
                        help='Architecture of the model')
    parser.add_argument("--exams", default=['J15', 'J30', 'M3', 'M12'], nargs='+', type=str,
                        help="Follow-ups exams to use")

    parser.add_argument("--output_dir", default="/pretrained_models", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", nargs='+', type=int,
                        help="Resolution size")
    parser.add_argument("--batch_size", default=24, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_every", default=1, type=int,
                        help="Run prediction on validation set every so many epochs.")

    parser.add_argument("--loss_margin", default=0, type=float,
                        help="Margin for cosine embedding loss") 
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_epochs", default=[100], type=int, nargs='+',
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_epochs", default=5, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--augmentation', default=0, type=int,
                    help='Data augmentation transforms')
    parser.add_argument('--dropout', default=0., type=float,
                    help='head dropout')    
    parser.add_argument('--normalize_feat', default=1, type=int,
                    help='Normalize features vectors')
    parser.add_argument("--curriculum", default=[0], nargs='+', type=int,
                        help="Curriculum training steps")

    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")

    parser.add_argument('--resume', default=0, type=int,
                    help='Resume training')
    parser.add_argument('--wandb_id', type=str,
                        help="Short run id")   
    parser.add_argument('--description', type=str,
                        help="Short run description (wandb name)")  
    args = parser.parse_args()
 
    args.name = args.wandb_id
    if not os.path.exists(os.path.join(args.output_dir, args.wandb_id)): 
        os.mkdir(os.path.join(args.output_dir, args.wandb_id)) 
    wandb.init(project="kidney_cnn",
               name=args.description,
               id=args.wandb_id,
               resume='allow')
    wandb.config.update(args)
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
                        level=logging.INFO)
    logger.warning("Devices: %s, n_gpu: %s" %(args.device, args.n_gpu))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
