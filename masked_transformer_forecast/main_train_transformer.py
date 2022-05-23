# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import time

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm

from models.vit import TimeSformer
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader_timesformer  
from torch.nn.modules.utils import _pair, _triple

import subprocess
from io import BytesIO
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import wandb
from skimage.transform import resize
import io

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

def save_ckp(args, ckp, is_best=False): 
    model_checkpoint = os.path.join(args.output_dir, 'checkpoint_timesformer', args.wandb_id, "%s_checkpoint.bin" % args.wandb_id)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
    torch.save(ckp, model_checkpoint)
    if is_best:
        model_checkpoint = os.path.join(args.output_dir, 'pretrained_timesformer', args.wandb_id, "%s_best.bin" % args.wandb_id)
        torch.save(ckp, model_checkpoint)
        logger.info("Saved best model checkpoint to [DIR: %s]", args.output_dir)

def load_ckp(args, model, optimizer, scheduler):
	checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint_timesformer', args.wandb_id, "%s_checkpoint.bin" % args.wandb_id))
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	scheduler.load_state_dict(checkpoint['scheduler'])
	return model, optimizer, scheduler, checkpoint['wandb_step'], checkpoint['global_step'], checkpoint['epoch_step'], checkpoint['best_loss']

def setup(args):
    # Prepare model
    if args.pretrained_dir:
        args.pretrained_dir = os.path.join('/gpfs/workdir/mileckil/data/pretrained_models/timesformer', args.pretrained_dir)
    model = TimeSformer(img_size=args.img_size, num_classes=1, num_frames=4, attention_type='divided_space_time', pretrained_model=args.pretrained_dir)

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

def test(args, model, test_loader, wandb_step, gfr_scaler):
    # Test!
    test_losses = AverageMeter()

    logger.info("\n\n***** Running Test *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)
    loss_fct = torch.nn.MSELoss()
    for step, batch in enumerate(epoch_iterator):
        with torch.no_grad():
            x1 = batch['J15']['data'].permute(0, 1, 4, 3, 2).float()
            x2 = batch['J30']['data'].permute(0, 1, 4, 3, 2).float()
            x3 = batch['M3']['data'].permute(0, 1, 4, 3, 2).float()
            x4 = batch['M12']['data'].permute(0, 1, 4, 3, 2).float()
            x = torch.stack((x1, x2, x3, x4), dim=2)
            y = batch['label'].float()
            x, y = x.to(args.device), y.to(args.device)
            yhat = model(x)
            test_loss = loss_fct(yhat, torch.unsqueeze(y.float(), dim=1))
            test_losses.update(test_loss.item())
        
        if len(all_preds) == 0:
            all_preds.append(yhat.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
 
        else:
            all_preds[0] = np.append(
                all_preds[0], yhat.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Testing... (loss=%2.5f)" % test_losses.val)
 
    all_preds, all_label = all_preds[0], all_label[0]
    all_preds = gfr_scaler.inverse_transform(all_preds.reshape(-1, 1))
    all_label = gfr_scaler.inverse_transform(all_label.reshape(-1, 1))
    test_r2 = r2_score(all_label, all_preds)
    test_rmse = np.sqrt(mean_squared_error(all_label, all_preds))

    logger.info("\n")
    logger.info("Test Results")
    logger.info("Test Loss: %2.5f" % test_losses.avg)
    logger.info("Test R2: %2.5f" % test_r2)
    logger.info("Test RMSE: %2.5f" % test_rmse)

    wandb.log({'test/loss': test_losses.avg, 
               'test/r2': test_r2, 
               'test/rmse': test_rmse}, step=wandb.run.step+wandb_step+1)
    
    return test_losses.avg

     
def valid(args, model, eval_loader, wandb_step, global_step, epoch_step, gfr_scaler):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("\n\n***** Running Validation *****")
    logger.info("  Num steps = %d", len(eval_loader))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(eval_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)
    loss_fct = torch.nn.MSELoss()
    for step, batch in enumerate(epoch_iterator):
        with torch.no_grad():
            x1 = batch['J15']['data'].permute(0, 1, 4, 3, 2).float()
            x2 = batch['J30']['data'].permute(0, 1, 4, 3, 2).float()
            x3 = batch['M3']['data'].permute(0, 1, 4, 3, 2).float()
            x4 = batch['M12']['data'].permute(0, 1, 4, 3, 2).float()
            x = torch.stack((x1, x2, x3, x4), dim=2)
            y = batch['label'].float()
            x, y = x.to(args.device), y.to(args.device)
            yhat = model(x)
            eval_loss = loss_fct(yhat, torch.unsqueeze(y.float(), dim=1))
            eval_losses.update(eval_loss.item())
        
        if len(all_preds) == 0:
            all_preds.append(yhat.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
 
        else:
            all_preds[0] = np.append(
                all_preds[0], yhat.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
 
    all_preds, all_label = all_preds[0], all_label[0]
    all_preds = gfr_scaler.inverse_transform(all_preds.reshape(-1, 1))
    all_label = gfr_scaler.inverse_transform(all_label.reshape(-1, 1))
    eval_r2 = r2_score(all_label, all_preds)
    eval_rmse = np.sqrt(mean_squared_error(all_label, all_preds))

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid R2: %2.5f" % eval_r2)
    logger.info("Valid RMSE: %2.5f" % eval_rmse)

    wandb.log({'validation/loss': eval_losses.avg, 
               'validation/r2': eval_r2, 
               'validation/rmse': eval_rmse, 
               'global_step': global_step, 
               'epoch_step': epoch_step}, step=wandb.run.step+wandb_step+1)
    
    return eval_losses.avg


def train(args, model):
    """ Train the model """
    # Prepare dataset
    train_loader, eval_loader, test_loader, gfr_scaler = get_loader_timesformer(args)

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
   
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    wandb_step, global_step, epoch_step, best_loss = 0, 0, 0, 1e12
    if args.resume:
        model, optimizer, scheduler, wandb_step, global_step, epoch_step, best_loss = load_ckp(args, model, optimizer, scheduler)
    while True:
        t = time.time()
        epoch_step += 1
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=False)
        loss_fct = torch.nn.MSELoss() 
        for step, batch in enumerate(epoch_iterator):
            x1 = batch['J15']['data'].permute(0, 1, 4, 3, 2).float()
            x2 = batch['J30']['data'].permute(0, 1, 4, 3, 2).float()
            x3 = batch['M3']['data'].permute(0, 1, 4, 3, 2).float()
            x4 = batch['M12']['data'].permute(0, 1, 4, 3, 2).float()
            x = torch.stack((x1, x2, x3, x4), dim=2)
            y = batch['label'].float()
            x, y = x.to(args.device), y.to(args.device)
            yhat = model(x)
            loss = loss_fct(yhat, torch.unsqueeze(y.float(), dim=1))
            loss.backward()

            losses.update(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val))
            if (time.time()-t)/60 > 120:
                 ckp = {'wandb_step': wandb.run.step,
               'global_step': global_step,
               'epoch_step': epoch_step,
               'best_loss': best_loss,
               'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict()}
                 save_ckp(args, ckp, is_best=False)
                 t = time.time()
            if global_step % t_total == 0:
                break

        if epoch_step % args.eval_every == 0:
            eval_loss = valid(args, model, eval_loader, wandb_step, global_step, epoch_step, gfr_scaler)
            if best_loss > eval_loss:
                save_ckp(args, model, is_best=True)
                best_loss = eval_loss
            model.train()
        if epoch_step in args.vis_atn:
            visualize_attention_pairs(args, model, eval_loader, wandb_step, global_step, epoch_step)
            model.train()
 
        wandb.log({'train/epoch_loss': losses.avg, 'global_step': global_step, 'epoch_step': epoch_step}, step=wandb.run.step+wandb_step+1)
        wandb.log({'train/lr': scheduler.get_last_lr()[0], 'global_step': global_step, 'epoch_step': epoch_step}, step=wandb.run.step+wandb_step+1) #get_last_lr()
        losses.reset()
        ckp = {'wandb_step': wandb.run.step,
               'global_step': global_step,
               'epoch_step': epoch_step,
               'best_loss': best_loss,
               'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict()}
        save_ckp(args, ckp, is_best=False)
        if global_step % t_total == 0:
            break
    test_loss = test(args, model, test_loader, wandb_step, gfr_scaler)
    logger.info("Best Eval Loss: \t%f" % best_loss)
    logger.info("Test Loss: \t%f" % test_loss)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--target", choices= ['pred_M12', 'pred_M24'], type=str,
                        help="Which upstream task.")
    parser.add_argument("--val_size", default=10, type=int,
                        help="Size of the testset")
    parser.add_argument("--exams", default=['J15', 'J30', 'M3', 'M12'], nargs='+', type=str,
                        help="follow-ups exams to use")
    parser.add_argument("--mri_series", choices= ['TUB', 'ALL'], type=str,
                        help="Which mri serires for training.")
    parser.add_argument("--max_exams_comb", default=0, type=int,
                        help="Number of exams series combinaison for each patient")

    parser.add_argument("--pretrained_dir", type=str, default=None,
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="/gpfs/workdir/mileckil/output", type=str,
                       help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=[224], nargs='+', type=int,
                        help="Resolution size")
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_every", default=1, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--vis_atn", default=[0], nargs='+', type=int,
                        help= "When visualization of attention maps")

    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_epochs", default=[100], type=int, nargs='+',
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--augmentation', default=0, type=int,
                    help='data augmentation transforms')

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--resume', default=0, type=int,
                    help='resume training')
    parser.add_argument('--wandb_id', type=str,
                        help="short run id")
    parser.add_argument('--description', type=str,
                        help="short run description (wandb name)")  
    args = parser.parse_args()
    
    args.name = args.wandb_id 
    if not os.path.exists(os.path.join(args.output_dir, 'pretrained_timesformer', args.wandb_id)):
        os.mkdir(os.path.join(args.output_dir, 'pretrained_timesformer', args.wandb_id))
        os.mkdir(os.path.join(args.output_dir, 'checkpoint_timesformer', args.wandb_id))
    wandb.init(project="kidney_timesformer",
               name=args.description,
               id=args.wandb_id,
               resume='allow')
    wandb.config.update(args)
    # Setup CUDA, GPU & distributed training
    print('Using GPU')
    device = "cuda"
    args.n_gpu = torch.cuda.device_count()
    args.device = device

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
