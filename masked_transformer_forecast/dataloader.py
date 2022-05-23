import logging
import sys
import psutil
import torch

#from torchvision import transforms, datasets
from torch.utils.data import DataLoader #, RandomSampler, DistributedSampler, SequentialSampler

# Kidney packages
import os
import numpy as np
import pandas as pd
import torchio as tio
from sklearn.model_selection import train_test_split, ParameterGrid
from pprint import pprint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

import sys
sys.path.insert(0, '../')
from  misc.data_import import get_patient_seq_paths

logger = logging.getLogger(__name__)

def tryconvert(value, default, *types):
    for t in types:
        try:
            return t(value)
        except (ValueError, TypeError):
            continue
    return default

def prepare_biological_df(path_to_biological_data):
    df=pd.read_csv(os.path.join(path_to_biological_data, 'Imagnct_Biological_Results_july2019.csv'), sep=';')
    df.columns =['IMAGNCT', 'DIVAT', 'Datetime', 'sCreatinine', 'uCreatinine','Proteinurinas']

    df['Datetime'] = df['Datetime'].apply(lambda x: x+' 00:00' if len(x)<12 else x)
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M')
    df['sCreatinine'] = df['sCreatinine'].apply(lambda x: tryconvert(x, None, int))
    df['uCreatinine'] = df['uCreatinine'].apply(lambda x: tryconvert(x, None, int))
    df['Proteinurinas'] = df['Proteinurinas'].apply(lambda x: tryconvert(x, None, int))
    return df

def prepare_inclusion_df(path_to_biological_data):
    df=pd.read_csv(os.path.join(path_to_biological_data, 'imag-nct_export-csv_20190722_FGN/1_INCLUSION.csv'), sep=';')
    df['dte_incl'] = pd.to_datetime(df['dte_incl'], format='%d/%m/%Y')
    df['ddn'] = pd.to_datetime(df['ddn'], format='%d/%m/%Y')
    df['dte_transplant'] = pd.to_datetime(df['dte_transplant'], format='%d/%m/%Y')
    df['age_transplant'] = df['dte_transplant'] - df['ddn']
    df['age_transplant'] = df['age_transplant'].apply(lambda x: x.total_seconds()/(60*60*24*365.25))
    df['taille'] = df['taille'].apply(lambda x: x/100 if not np.isnan(x) and x>100 else x)
    df[['complic_chirtypec1',
        'complic_chirtypec2',
        'complic_chirtypec3',
        'complic_chirtypec4',
        'complic_chirtypec5',
        'complic_chirtypec6',
        'complic_chirtypec7']] = df[['complic_chirtypec1',
                                     'complic_chirtypec2',
                                     'complic_chirtypec3',
                                     'complic_chirtypec4',
                                     'complic_chirtypec5',
                                     'complic_chirtypec6',
                                     'complic_chirtypec7']].fillna(value=0)
    return df


def prepare_biological_data(df_biological, df_inclusion, exams, pred_dates, path_to_data, normalize=True):
    # biological data
    df_biological = df_biological.drop(columns = ['DIVAT', 'uCreatinine', 'Proteinurinas'])
    df_biological.rename(columns = {'IMAGNCT':'patient'}, inplace = True)

    # inclusion data
    df_inclusion = df_inclusion[['patient', 'dte_transplant']]

    df_exams = pd.read_csv(os.path.join(path_to_data, 'df_save_info_bold_t2_alldce.csv'))
    for exam in list(exams.keys())[1:]:
        df_exams[exam] = pd.to_datetime(df_exams[exam], format='%Y-%m-%d')
    df_biological = df_biological.merge(df_exams[['patient']+list(exams.keys())[1:]], how='outer', on='patient')

    df_biological = df_biological.merge(df_inclusion, how='outer', on='patient')
    df_biological = df_biological.dropna(subset=['sCreatinine'])
    df_biological['days_after_transplant'] = (df_biological['Datetime'] - df_biological['dte_transplant'])
    df_biological['days_after_transplant'] = df_biological['days_after_transplant'].apply(lambda x: x.total_seconds()/86400)
    for exam in list(exams.keys())[1:]:
        df_biological[exam] = (df_biological[exam] - df_biological['dte_transplant'])
        df_biological[exam] = df_biological[exam].apply(lambda x: x.total_seconds()/86400)
    for pred_date in list(pred_dates.keys()):
        df_biological[pred_date] = pred_dates[pred_date]
    df_biological['J0'] = -1
    df_biological = df_biological.fillna(exams)

    if normalize:
        clip_value = df_biological['sCreatinine'].quantile(0.95)
        df_biological[['sCreatinine']] = df_biological[['sCreatinine']].clip(upper=clip_value)
        gfr_scaler = make_pipeline(StandardScaler(), MinMaxScaler())
        df_n = df_biological[['sCreatinine']]
        gfr_scaler.fit(df_n)
        scaled_data = gfr_scaler.transform(df_n)
        df_n = pd.DataFrame(scaled_data, columns=df_n.columns, index=df_n.index)
        df_biological['sCreatinine'] = df_n
        return df_biological, gfr_scaler, clip_value
    else:
        return df_biological, None, None

def get_loader_timesformer(args):
	data_settings = {
		'path_to_data': '/gpfs/workdir/mileckil/data/data_kidney_mri',
		'path_to_biological_data': '/gpfs/workdir/mileckil/data/clinicobiological_data',
		'path_to_dataframes': '/gpfs/workdir/mileckil/data/dataframes',
		'key_words_seqs': [['TUB', 'tub', 'WATER', 'AX', 'LAVA', ], ['WATER', 'AX', 'LAVA']],
		'mins_key_words': [4, 3],
		'exams': args.exams,
		'mri_filename': 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz',
		}
	df_biological = prepare_biological_df(data_settings['path_to_biological_data'])
	df_inclusion = prepare_inclusion_df(data_settings['path_to_biological_data'])

	exams = {'J0':-1, 'J15': 15, 'J30': 30, 'M3':91, 'M12': 365}
	pred_dates = {'pred_J15': 365+15, 'pred_J30': 365+30, 'pred_M3':365+91, 'pred_M12': 365+365}

	df_biological, gfr_scaler, clip_value = prepare_biological_data(df_biological, df_inclusion, exams, pred_dates, data_settings['path_to_dataframes'], normalize=True)

	gfr_data = {}

	for patient in np.unique(df_biological['patient']):
		gfr_data[patient] = {}
		last_exam_date = -2000
		for exam in list(exams.keys()) + list(pred_dates.keys()):
			df = df_biological[(df_biological['patient']==patient) & (last_exam_date<df_biological['days_after_transplant']) & (df_biological['days_after_transplant']<=df_biological[exam])]
			gfr_data[patient][exam] = np.stack((df['days_after_transplant'].values, df['sCreatinine'].values))
			last_exam_date = df_biological[exam]
		df = df_biological[(df_biological['patient']==patient) & (300<=df_biological['days_after_transplant'])]
		gfr_data[patient]['pred'] = np.stack((df['days_after_transplant'].values, df['sCreatinine'].values))

	preds = {'pred_M6':(12+6)*30.5, 'pred_M12':(12+12)*30.5, 'pred_M18':(12+18)*30.5, 'pred_M24':(12+24)*30.5, 'pred_M30':(12+30)*30.5, 'pred_M36':(12+36)*30.5}
	preds_dist_thresold = 61

	gfr_preds = {}
	for patient in gfr_data.keys():
		gfr_preds[patient] = {}
		for pred_date in preds.keys():
			if gfr_data[patient]['pred'].shape[1]==0:
				gfr_preds.pop(patient, None)
				#gfr_data_stats.pop(patient, None)
			else:
				dist_to_pred = abs(gfr_data[patient]['pred'][0]-preds[pred_date])
				idx = np.where(dist_to_pred<preds_dist_thresold)
				if len(idx[0]) != 0:
					gfr_preds[patient][pred_date] = np.mean(gfr_data[patient]['pred'][1][idx])
				else:
					gfr_preds[patient][pred_date] = np.nan

	thresholds = [110]
	n_labels = len(thresholds)
	thresholds = gfr_scaler.transform(np.array(thresholds).reshape(-1, 1))
	def value_to_class(value, thresholds):
		lst = list(thresholds).copy()
		lst.append(value)
		lst.sort()
		#lst.reverse()
		return lst.index(value)

	gfr_class_preds = {}
	for patient in gfr_preds.keys():
		gfr_class_preds[patient] = {}
		for pred_date in gfr_preds[patient].keys():
			if ~np.isnan(gfr_preds[patient][pred_date]):
				gfr_class_preds[patient][pred_date] = value_to_class(gfr_preds[patient][pred_date], thresholds)

	target_patients = set(np.array(list(gfr_preds.keys()))[(~np.isnan(np.stack([gfr_preds[patient][args.target] for patient in gfr_preds.keys()])))])
	mri_patients = next(os.walk(os.path.join(data_settings['path_to_data'])))[1]
	patients = list(set(mri_patients) & target_patients)
	dataset = {}
	for patient in patients:
		dataset[patient] = {}
		for exam in data_settings['exams']:
			if os.path.exists(os.path.join(data_settings['path_to_data'], patient, exam)):
				path_to_volumes = get_patient_seq_paths(data_settings['path_to_data'], exam, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=patient)
				if path_to_volumes:
					dataset[patient][exam] = {}
					for seq_key in path_to_volumes.keys():
						if os.path.exists(os.path.join(path_to_volumes[seq_key], 'preprocessed')):
							dataset[patient][exam][seq_key] = path_to_volumes[seq_key]
	test_patients = ['001-0070-J-A','001-0092-B-J','001-0102-M-L','001-0036-R-E','001-0106-N-C','001-0096-S-M','001-0050-S-J','001-0017-C-W','001-0052-B-A','001-0015-S-K','001-0023-D-D','001-0079-T-M','001-0029-C-S','001-0080-K-D','001-0099-V-V','001-0040-D-D','001-0119-R-A','001-0075-D-A','001-0018-C-A','001-0004-P-R']
	train_patients = list(set(patients)-set(test_patients))
	train_patients, val_patients = train_test_split(train_patients, test_size=args.val_size, stratify=[gfr_class_preds[patient][args.target] for patient in train_patients]) 

	if args.augmentation==1:
		train_transform = tio.Compose([tio.RandomFlip(),
					       tio.RandomAffine(p=0.5)])
		val_transform = tio.Compose([])
	elif args.augmentation==2:
		train_transform = tio.Compose([tio.RandomFlip(),
					       tio.RandomAffine(p=0.5),
					       tio.RandomBlur((0,0.5)),
					       tio.RandomNoise(0, (0.05))])
		val_transform = tio.Compose([])
	else:   
		train_transform = tio.Compose([])
		val_transform = tio.Compose([])

	if args.mri_series == 'ALL':
		train_subjects = []
		for patient in train_patients:
			exam_grid = {}
			for exam in args.exams:
				if exam in dataset[patient]:
					exam_grid[exam] = list(dataset[patient][exam].keys())
				else:
					exam_grid[exam] = [None]
			if args.max_exams_comb == 0:
				exams_comb = list(ParameterGrid(exam_grid))
			else:
				exams_comb = len(list(ParameterGrid(exam_grid)))
				exams_comb = np.random.choice(list(ParameterGrid(exam_grid)), size=min(args.max_exams_comb, exams_comb))
			for param in exams_comb:
				subject_dict = {'label': gfr_preds[patient][args.target]}
				for exam in args.exams:
					if param[exam] is None:
						subject_dict[exam] = tio.ScalarImage(os.path.join(data_settings['path_to_data'], 'dummy_mri.nii.gz'))
					else:
						image_path = os.path.join(dataset[patient][exam][param[exam]], data_settings['mri_filename'])
						subject_dict[exam] = tio.ScalarImage(image_path)
				subject = tio.Subject(subject_dict)
				train_subjects.append(subject)
		train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform, load_getitem=False)
	elif args.mri_series == 'TUB':	
		train_subjects = []
		for patient in train_patients:
			subject_dict = {'label': gfr_preds[patient][args.target]}
			for exam in args.exams:
				if ~(exam in dataset[patient]):
					subject_dict[exam] = tio.ScalarImage(os.path.join(data_settings['path_to_data'], 'dummy_mri.nii.gz'))
				else:   
					image_path = os.path.join(dataset[patient][exam]['TUB'], data_settings['mri_filename'])
					subject_dict[exam] = tio.ScalarImage(image_path)
			subject = tio.Subject(subject_dict)
			train_subjects.append(subject)
		train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform, load_getitem=False)

	val_subjects = []
	for patient in val_patients:
		subject_dict = {'label': gfr_preds[patient][args.target]}
		for exam in args.exams:
			if ~(exam in dataset[patient]):
				subject_dict[exam] = tio.ScalarImage(os.path.join(data_settings['path_to_data'], 'dummy_mri.nii.gz'))
			else:
				image_path = os.path.join(dataset[patient][exam]['TUB'], data_settings['mri_filename'])
				subject_dict[exam] = tio.ScalarImage(image_path)
		subject = tio.Subject(subject_dict)
		val_subjects.append(subject)
	val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)

	test_subjects = []
	for patient in test_patients:
		subject_dict = {'label': gfr_preds[patient][args.target]}
		for exam in args.exams:
			if ~(exam in dataset[patient]):
				subject_dict[exam] = tio.ScalarImage(os.path.join(data_settings['path_to_data'], 'dummy_mri.nii.gz'))
			else:
				image_path = os.path.join(dataset[patient][exam]['TUB'], data_settings['mri_filename'])
				subject_dict[exam] = tio.ScalarImage(image_path)
		subject = tio.Subject(subject_dict)
		test_subjects.append(subject)
	test_dataset = tio.SubjectsDataset(test_subjects, transform=val_transform)

	print('Length train set', len(train_dataset))
	print('Length val set', len(val_dataset))
	print('Length test set', len(test_dataset))
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
	validation_loader = DataLoader(val_dataset, batch_size=args.batch_size)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

	return train_loader, validation_loader, test_loader, gfr_scaler

