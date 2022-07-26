import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
import shutil
import json
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from scipy.special import softmax
 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, '../')
from models.transformer import TransformerEncoder
from models.lstm import LSTMEncoder
from dataloader import sCreat_Dataset, MRI_Dataset 
from data_imports import *
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.metrics import AverageMeter, acc, f1, recall, precision, roc_auc
from utils.misc import set_seed

import warnings
warnings.filterwarnings('ignore') 

parser = argparse.ArgumentParser(description='Kidney features attention')
parser.add_argument("--input", choices=['sCreat', 'img', 'radiomics'], type=str)
parser.add_argument("--target", type=str)
parser.add_argument("--pretraining", type=str)
parser.add_argument("--model", choices=['tsf', 'lstm'], type=str)

parser.add_argument("--normalize", type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--n_heads", default=1, type=int)
parser.add_argument("--n_layers", default=1, type=int)
parser.add_argument("--ffwd_dim", default=64, type=int)
parser.add_argument("--dropout", default=0., type=float)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--warmup_epochs", default=10, type=int)
parser.add_argument("--decay_type", default='cosine', type=str)
parser.add_argument("--augmentation", type=int)
parser.add_argument("--n_splits", default=3, type=int)
parser.add_argument("--weight_decay", default=0, type=float)

parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument('--description', type=str, help="short run description (wandb name)")
args = parser.parse_args()
print(args)
# set wandb
wandb.init(project='kidney_features_attention',
           name=args.description)
wandb.config.update(args)

# set random seed
set_seed(args)

# Setup CUDA, GPU & distributed training
print('Using GPU')
device = "cuda"
args.n_gpu = torch.cuda.device_count()
args.device = device


path_to_biological_data = '../data/dummy_dataframes'
path_to_features = '../data/dummy_features'
path_to_models = './save/temp_models/' + args.description
if not os.path.exists(path_to_models):
    os.mkdir(path_to_models)

exams = {'D0':-1, 'D15': 15, 'D30': 30, 'M3':91, 'M12': 365}
pred_dates = {'pred_D15': 365+15, 'pred_D30': 365+30, 'pred_M3':365+91, 'pred_M12': 365+365}
seq_len = len(exams)-1 

if args.input == 'radiomics':
	dict_features = get_radiomics_features(args, path_to_features, list(exams.keys())[1:])
elif args.input == 'img':
	dict_features = get_contrastive_features(args, path_to_features, list(exams.keys())[1:])

df_sCreat, sCreat_scaler, clip_value = prepare_biological_data(args, path_to_biological_data)

sCreat_data = get_sCreat_data_per_followup(df_sCreat, exams, pred_dates)
sCreat_data_stats, sCreat_masks = get_sCreat_features(sCreat_data, exams, pred_dates, seq_len)

if args.input == 'img' or args.input == 'radiomics':
	feat_masks = get_features_masks(dict_features, sCreat_data_stats, exams, seq_len)

if args.model == 'tsf':
	filled_masks = None
elif args.model == 'lstm':
	if args.input == 'img':
		with open(os.path.join(path_to_features, "dummy_filled_masks.json")) as json_file:
			filled_masks = json.load(json_file)
	elif args.input == 'radiomics':
		with open(os.path.join(path_to_features, "dummy_filled_masks.json")) as json_file:
			filled_masks = json.load(json_file)

label_dates = {'pred_M6':(12+6)*30.5, 'pred_M12':(12+12)*30.5, 'pred_M18':(12+18)*30.5, 'pred_M24':(12+24)*30.5, 'pred_M30':(12+30)*30.5, 'pred_M36':(12+36)*30.5}
sCreat_labels = get_sCreat_labels(sCreat_data, label_dates, preds_dist_thresold=61)
sCreat_class_labels, n_labels = sCreat_labels_to_classes(sCreat_labels, sCreat_scaler, thresholds=[110])

if args.input == 'sCreat':
	patients, input_size = get_available_patients_sCreat(args, sCreat_data_stats, sCreat_labels, sCreat_masks, min_seq_len=3)
elif args.input == 'img' or args.input == 'radiomics':
	patients, input_size = get_available_patients_features(args, dict_features, sCreat_labels, exams)

best_loss = np.zeros(args.n_splits)
best_f1 = np.zeros(args.n_splits)
best_roc_auc = np.zeros(args.n_splits)
best_precision = np.zeros(args.n_splits)
best_recall = np.zeros(args.n_splits)

test_set = ['001-0001-A-A', '001-0002-B-B']
train_set = list(set(patients)-set(test_set))
labels = np.array([sCreat_class_labels[patient][args.target] for patient in train_set])
class_weight = np.sum(labels==0) / np.sum(labels==1)

train_set, test_set = np.array(train_set), np.array(test_set)
skf = StratifiedKFold(n_splits=args.n_splits)
sm = nn.Sigmoid()

for it_cv, (train_index, val_index) in enumerate(skf.split(train_set, [sCreat_class_labels[patient][args.target] for patient in train_set])):
	print('####### Fold {}/{} #######'.format(it_cv+1, args.n_splits))
	train_patients, val_patients = train_set[train_index], train_set[val_index]

	if args.input == 'sCreat': 
		train_dataset = sCreat_Dataset(train_patients, sCreat_data_stats, sCreat_masks,  
						list(exams.keys())[1:], pred_dates.keys(), sCreat_class_labels, label_date=args.target)
		val_dataset = sCreat_Dataset(val_patients, sCreat_data_stats, sCreat_masks,  
						list(exams.keys())[1:], pred_dates.keys(), sCreat_class_labels, label_date=args.target)
		train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)   
		val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)     
	elif args.input == 'img' or args.input == 'radiomics':
		train_dataset = MRI_Dataset(train_patients, dict_features, feat_masks, sCreat_data_stats, sCreat_masks, input_size,
						list(exams.keys())[1:], pred_dates.keys(), sCreat_class_labels, train=True, label_date=args.target, filled_masks=filled_masks)
		val_dataset = MRI_Dataset(val_patients, dict_features, feat_masks, sCreat_data_stats, sCreat_masks, input_size,
						list(exams.keys())[1:], pred_dates.keys(), sCreat_class_labels, train=False, label_date=args.target, filled_masks=filled_masks)
		train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
		val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

	criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weight)).to(device)
	if args.model == 'tsf': 
		model = TransformerEncoder(seq_len, input_size, args.n_heads, args.n_layers, args.ffwd_dim, n_labels=n_labels, dropout=args.dropout, avg_pool=False).to(device)
		for p in model.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)
	elif args.model == 'lstm':
		model = LSTMEncoder(seq_len, input_size, args.n_layers, args.ffwd_dim, n_labels=n_labels, dropout=args.dropout, avg_pool=False).to(device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)   
	num_steps = args.epochs * len(train_loader)
	warmup_steps = args.warmup_epochs * len(train_loader) 
	if args.decay_type == "cosine":
		scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_steps)
	else:
		scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_steps)


	best_val = 1e12
	for epoch in range(1,args.epochs+1):
		## TRAINING ##
		model.train()
		train_losses = AverageMeter()
		outputs, preds, lbls = None, None, None

		for i, batch, in enumerate(train_loader):
			optimizer.zero_grad()
			src, tgt, src_mask, tgt_mask, lbl,_ = batch
			src, src_mask, lbl = src.to(device), src_mask.to(device), lbl.to(device)
			if args.model == 'tsf':
				output = model(src, src_key_padding_mask=src_mask)
			elif args.model == 'lstm':
				output = model(src, device)
			loss=criterion(output, torch.unsqueeze(lbl.float(), dim=1))
			loss.backward()
			optimizer.step()
			scheduler.step()
			train_losses.update(loss.item())
			pred = (sm(output)>0.5)
			output = sm(output)
			if outputs is None:
				outputs = output.detach().cpu().numpy()
				preds = pred.detach().cpu().numpy()
				lbls = lbl.long().detach().cpu().numpy()
			else:
				outputs = np.concatenate((outputs, output.detach().cpu().numpy()))
				preds = np.concatenate((preds, pred.detach().cpu().numpy()))
				lbls = np.concatenate((lbls, lbl.long().detach().cpu().numpy()))
		
		train_precision = precision(lbls, preds)
		train_recall = recall(lbls, preds)
		train_f1 = f1(lbls, preds)
		train_roc_auc = roc_auc(lbls, outputs)

		wandb.log({'train/loss_cv_{}'.format(it_cv): train_losses.avg, 
			'train/f1_cv_{}'.format(it_cv): train_f1, 
			'train/roc_auc_cv_{}'.format(it_cv): train_roc_auc, 
			'train/precision_cv_{}'.format(it_cv): train_precision,
			'train/recall_cv_{}'.format(it_cv): train_recall,
			'epoch_step': epoch})

		## VALIDATION ##
		model.eval()
		eval_losses = AverageMeter()
		outputs, preds, lbls = None, None, None
				   
		for i, batch, in enumerate(val_loader):
			src, tgt, src_mask, tgt_mask, lbl,_ = batch
			src, src_mask, lbl = src.to(device), src_mask.to(device), lbl.to(device)
			if args.model == 'tsf':
				output = model(src, src_key_padding_mask=src_mask)
			elif args.model == 'lstm':
				output = model(src, device)
			loss=criterion(output, torch.unsqueeze(lbl.float(), dim=1))
			pred = (sm(output)>0.5)
			output = sm(output)

			eval_losses.update(loss.item())
			if outputs is None:
				outputs = output.detach().cpu().numpy()
				preds = pred.detach().cpu().numpy()
				lbls = lbl.long().detach().cpu().numpy()
			else:
				outputs = np.concatenate((outputs, output.detach().cpu().numpy()))
				preds = np.concatenate((preds, pred.detach().cpu().numpy()))
				lbls = np.concatenate((lbls, lbl.long().detach().cpu().numpy()))

		val_precision = precision(lbls, preds)
		val_recall = recall(lbls, preds)
		val_f1 = f1(lbls, preds)
		val_roc_auc = roc_auc(lbls, outputs)

		wandb.log({'validation/loss_cv_{}'.format(it_cv): eval_losses.avg, 
		       'validation/f1_cv_{}'.format(it_cv): val_f1, 
		       'validation/roc_auc_cv_{}'.format(it_cv): val_roc_auc, 
		       'validation/precision_cv_{}'.format(it_cv): val_precision,
		       'validation/recall_cv_{}'.format(it_cv): val_recall,
		       'epoch_step': epoch})

		if best_val >= eval_losses.avg:
			best_val = eval_losses.avg
			best_loss[it_cv] = eval_losses.avg
			best_f1[it_cv] = val_f1 
			best_roc_auc[it_cv] = val_roc_auc
			best_precision[it_cv] = val_precision
			best_recall[it_cv] = val_recall
			checkpoint = {"state_dict": model.state_dict()}
			torch.save(checkpoint, os.path.join(path_to_models, 'model_cv_{}'.format(it_cv)))

if args.input == 'sCreat':
	test_dataset = sCreat_Dataset(test_set, sCreat_data_stats, sCreat_masks,
					list(exams.keys())[1:], pred_dates.keys(), sCreat_class_labels, label_date=args.target)
	test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
elif args.input == 'img' or args.input == 'radiomics':
	test_dataset = MRI_Dataset(test_set, dict_features, feat_masks, sCreat_data_stats, sCreat_masks, input_size,
					list(exams.keys())[1:], pred_dates.keys(), sCreat_class_labels, train=False, label_date=args.target, filled_masks=filled_masks)
	test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

cv_outputs = None 
for it_cv in range(args.n_splits):
	if best_roc_auc[it_cv] > 0.5:
		checkpoint = torch.load(os.path.join(path_to_models, 'model_cv_{}'.format(it_cv)))
		model.load_state_dict(checkpoint['state_dict'])
		model.eval()
		outputs, preds, lbls, patients_list = None, None, None, None
		for i, batch, in enumerate(test_loader):
			src, tgt, src_mask, tgt_mask, lbl, p = batch
			src, src_mask, lbl = src.to(device), src_mask.to(device), lbl.to(device)
			if args.model == 'tsf':
				output = model(src, src_key_padding_mask=src_mask)
			elif args.model == 'lstm':
				output = model(src, device)
			output = sm(output)
			if outputs is None:
				outputs = output.detach().cpu().numpy()
				lbls = lbl.long().detach().cpu().numpy()
				patients_list = np.array(list(p))
			else:   
				outputs = np.concatenate((outputs, output.detach().cpu().numpy()))
				lbls = np.concatenate((lbls, lbl.long().detach().cpu().numpy()))
				patients_list = np.concatenate((patients_list, np.array(list(p))))
		if cv_outputs is None:
			cv_outputs = np.expand_dims(outputs, axis=0)
		else:   
			cv_outputs = np.concatenate((cv_outputs, np.expand_dims(outputs, axis=0)))

cv_outputs_mean = cv_outputs.mean(0)
cv_outputs_median = np.median(cv_outputs, axis=0)
cv_preds_mean = (cv_outputs_mean>0.5).astype('int32')
cv_preds_median = (cv_outputs_median>0.5).astype('int32')

test_f1_mean = f1(lbls, cv_preds_mean)
test_roc_auc_mean = roc_auc(lbls, cv_outputs_mean)
test_precision_mean = precision(lbls, cv_preds_mean)
test_recall_mean = recall(lbls, cv_preds_mean)
test_f1_median = f1(lbls, cv_preds_median)
test_roc_auc_median = roc_auc(lbls, cv_outputs_median)
test_precision_median = precision(lbls, cv_preds_median)
test_recall_median = recall(lbls, cv_preds_median)

df = pd.DataFrame(data=np.transpose(np.stack((patients_list, lbls, cv_preds_mean[:,0], cv_preds_median[:,0], cv_outputs_mean[:,0], cv_outputs_median[:,0]))), columns=['patient', 'label', 'pred_mean', 'pred_median', 'prob_mean', 'prob_median'])
df.to_csv('./save/results.csv')

wandb.log({'mean_cv/loss_mean': np.mean(best_loss), 'mean_cv/f1_mean': np.mean(best_f1), 'mean_cv/roc_auc_mean': np.mean(best_roc_auc), 'mean_cv/precision_mean': np.mean(best_precision), 'mean_cv/recall_mean': np.mean(best_recall), 
'mean_cv/loss_std': np.std(best_loss), 'mean_cv/f1_std': np.std(best_f1), 'mean_cv/roc_auc_std': np.std(best_roc_auc), 'mean_cv/precision_std': np.std(best_precision), 'mean_cv/recall_std': np.std(best_recall)})

wandb.log({'test/f1_mean': test_f1_mean, 'test/f1_median': test_f1_median, 
'test/roc_auc_mean': test_roc_auc_mean, 'test/roc_auc_median': test_roc_auc_median,
'test/precision_mean': test_precision_mean, 'test/precision_median': test_precision_median,
'test/recall_mean': test_recall_mean, 'test/recall_median': test_recall_median})

shutil.rmtree(path_to_models)
