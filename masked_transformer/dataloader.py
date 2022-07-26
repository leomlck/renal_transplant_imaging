import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

class sCreat_Dataset(data.Dataset):
    """
    Pytorch dataset class for source (and target) sCreat features sequences. Also get labels.
    """
    def __init__(self, patients, sCreat_data_stats, masks, exams, pred_dates, sCreat_class_preds, label_date='pred_M12'):
        self.patients = patients
        self.sCreat_data_stats = sCreat_data_stats
        self.exams = exams
        self.pred_dates = pred_dates
        self.masks = masks
        self.sCreat_class_preds = sCreat_class_preds
        self.label_date = label_date

    def __getitem__(self, index):
        patient = self.patients[index]
        dta = self.sCreat_data_stats[patient]
        src = np.stack([dta[exam] for exam in self.exams])
        src_mask = self.masks[patient]['src']
        tgt = np.stack([dta[pred_date] for pred_date in self.pred_dates])
        tgt_mask = self.masks[patient]['tgt']
        lbl = self.sCreat_class_preds[patient][self.label_date]
        return torch.tensor(src, dtype=torch.float), torch.tensor(tgt, dtype=torch.float), torch.tensor(src_mask, dtype=torch.bool), torch.tensor(tgt_mask, dtype=torch.bool), torch.tensor(lbl, dtype=torch.float), patient

    def __len__(self):
        return len(self.patients)

class MRI_Dataset(data.Dataset):
    """
    Pytorch dataset class for source (and target) imaging features sequences. Also get labels.
    If train==True, randomly fetch augmented features samples.
    For LSTM model, filled_masks has to be specified to know from which exam should we replace missing exams. (nearest available).
    """
    def __init__(self, patients, features, feat_masks, sCreat_data_stats, pred_masks, feat_size, exams, pred_dates, sCreat_class_preds, 
                 train=True, label_date='pred_M12', filled_masks=None):
        self.patients = patients
        self.features = features
        self.feat_masks = feat_masks
        self.sCreat_data_stats = sCreat_data_stats
        self.train = train
        self.label_date = label_date
        self.pred_masks = pred_masks
        self.pred_dates = pred_dates
        self.exams = exams
        self.sCreat_class_preds = sCreat_class_preds
        self.filled_masks = filled_masks
        self.feat_size = feat_size

    def __getitem__(self, index):
        patient = self.patients[index]
        dta = self.sCreat_data_stats[patient]
        src = np.zeros((4, self.feat_size))
        src_mask = self.feat_masks[patient]
        for i_exam, exam in enumerate(self.exams):
            if not src_mask[i_exam+1]:
                df = self.features[exam]
                if self.train:
                    aug_feats = df[(df.patient==patient)].drop(columns=['patient', 'aug']).values
                    src[i_exam] = aug_feats[random.randrange(0, aug_feats.shape[0])]
                else:
                    src[i_exam] = df[(df.patient==patient) & (df.aug==0)].drop(columns=['patient', 'aug']).values[0]
            elif self.filled_masks is not None:
                exm = self.filled_masks[patient][i_exam]
                df = self.features[exm]
                if self.train:
                    aug_feats = df[(df.patient==patient)].drop(columns=['patient', 'aug']).values
                    src[i_exam] = aug_feats[random.randrange(0, aug_feats.shape[0])]
                else:
                    src[i_exam] = df[(df.patient==patient) & (df.aug==0)].drop(columns=['patient', 'aug']).values[0] 
        tgt = np.stack([dta[pred_date] for pred_date in self.pred_dates])
        tgt_mask = self.pred_masks[patient]['tgt']
        lbl = self.sCreat_class_preds[patient][self.label_date]
        return torch.tensor(src, dtype=torch.float), torch.tensor(tgt, dtype=torch.float), torch.tensor(src_mask, dtype=torch.bool), torch.tensor(tgt_mask, dtype=torch.bool), torch.tensor(lbl, dtype=torch.float), patient

    def __len__(self):
        return len(self.patients)


