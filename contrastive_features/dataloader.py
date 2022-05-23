import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

import os
import numpy as np
import pandas as pd
import torchio as tio
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '../')
from  misc.data_import import get_patient_seq_paths

class CustomSampler(torch.utils.data.sampler.Sampler):
	"""
	Custom sampler to fetch all the positive sample pairs with a probability 1 and the remaining negative sample pairs with a uniform probability from all the available negative samples pool
	"""
        def __init__(self, dataset, num_samples):
                self.indices = list(range(len(dataset)))
                self.num_samples = len(self.indices) if num_samples is None else num_samples
                self.labels = np.array([subject.label for subject in dataset.dry_iter()]).astype('float32')
                sampler_probs = self.labels.copy().astype('float32')
                sampler_probs[np.where(sampler_probs<0)] = 1/len(np.where(sampler_probs<0)[0])
                self.sampler_probs = torch.DoubleTensor(sampler_probs)

        def __iter__(self):
                idxs = np.array([self.indices[i] for i in torch.multinomial(self.sampler_probs, self.num_samples, replacement=False)])
                labels = np.array([self.labels[i] for i in idxs])
                pos = np.where(labels==1)[0]
                neg = np.where(labels==-1)[0]
                ordered_idxs = []
                for i in range(len(pos)):
                        pos_neg_neg = np.concatenate(([idxs[pos[i]]], idxs[(neg[3*i: 3*i+3])]))
                        ordered_idxs = np.concatenate((ordered_idxs,pos_neg_neg))
                return (int(ordered_idxs[i]) for i in range(len(ordered_idxs)))

        def __len__(self):
                return self.num_samples

def get_loader_kidney_pairs(args, curriculum):
	"""
	Building dataset and dataloader.
	"""
	data_settings = {
		'path_to_data': '/gpfs/workdir/mileckil/data/data_kidney_mri',
		'path_to_targets': '/gpfs/workdir/mileckil/data/dataframes',
		'key_words_seqs': [['TUB', 'tub', 'WATER', 'AX', 'LAVA', ], ['WATER', 'AX', 'LAVA']],
		'mins_key_words': [4, 3],
		'exams': args.exams,
		'mri_filename': 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz',
		}

	df_targets = pd.read_csv(os.path.join(data_settings['path_to_targets'], 'df_targets.csv'), sep=';')
	if 'Unnamed: 0' in df_targets.columns:
		print('\nUnnamed 0 column removed')
		df_targets = df_targets.drop(columns='Unnamed: 0')
	df_targets = df_targets[['patient', args.target]]

	# Fetch all the available exams with the corresponding clinical information value	
	dataset = []
	patients = next(os.walk(os.path.join(data_settings['path_to_data'])))[1]
	for patient in patients:
		for exam in data_settings['exams']:
			if os.path.exists(os.path.join(data_settings['path_to_data'], patient, exam)):
				path_to_volumes = get_patient_seq_paths(data_settings['path_to_data'], exam, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=patient)
				if path_to_volumes:
					try:
						val = df_targets.loc[df_targets['patient']==patient][args.target].values[0]
					except (KeyError, IndexError):
						val = np.nan
					if not np.isnan(val):
						for key in path_to_volumes.keys():
							if os.path.exists(os.path.join(path_to_volumes[key], 'preprocessed')):
								dataset_id = patient + '_' + exam + '_' + key
								dataset.append((dataset_id, val))
	# Build a data of pairs, set label y of the pair according to the difference en clinical information value and a fixed threshold
	X, y = [], []
	dataset2 = dataset.copy()
	for data1 in dataset:
		dataset2.remove(data1)
		for data2 in dataset2:
			if (data1[0].split('_')[0] != data2[0].split('_')[0]):
				if abs(data1[1]-data2[1]) < curriculum[0]:
					X.append([data1[0], data2[0]])			
					y.append(1)
				elif abs(data1[1]-data2[1]) >= curriculum[1]:
					X.append([data1[0], data2[0]])
					y.append(-1)

	X, y = np.array(X), np.array(y)

	# Fix the dataset size, the positive sample pairs (25%) and make a pool of negative samples
	nb_pos_ex = len(np.where(y==1)[0])
	pos_ex = np.where(y==1)
	neg_ex = np.where(y==-1)
	neg_ex_to_keep = np.random.choice(neg_ex[0], size=10*args.dataset_size, replace=False)
	neg_ex_to_keep_val = np.random.choice(neg_ex_to_keep, size=3*args.testset_size//4, replace=False)
	neg_ex_to_keep_train = np.setdiff1d(neg_ex_to_keep, neg_ex_to_keep_val)
	pos_ex_to_keep = np.random.choice(pos_ex[0], size=args.dataset_size//4, replace=False)
	pos_ex_to_keep_val = np.random.choice(pos_ex_to_keep, size=args.testset_size//4, replace=False)
	pos_ex_to_keep_train = np.setdiff1d(pos_ex_to_keep, pos_ex_to_keep_val)
	to_keep_train = (np.concatenate((pos_ex_to_keep_train, neg_ex_to_keep_train)),)
	to_keep_val = (np.concatenate((pos_ex_to_keep_val, neg_ex_to_keep_val)),)
	X_train, X_val, y_train, y_val = X[to_keep_train], X[to_keep_val], y[to_keep_train], y[to_keep_val]       

	# Data augmentations
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

	# Build dataset pytorch objects from TorchIO package
	train_subjects = []
	for pair, label in zip(X_train, y_train):
		ID_patient1 = pair[0].split('_')[0]
		ID_exam1 = pair[0].split('_')[1]
		ID_key1 = pair[0].split('_')[2]
		path_to_volumes1 = get_patient_seq_paths(data_settings['path_to_data'], ID_exam1, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient1)
		image_path1 = os.path.join(path_to_volumes1[ID_key1], data_settings['mri_filename'])

		ID_patient2 = pair[1].split('_')[0]
		ID_exam2 = pair[1].split('_')[1]
		ID_key2 = pair[1].split('_')[2]
		path_to_volumes2 = get_patient_seq_paths(data_settings['path_to_data'], ID_exam2, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient2)
		image_path2 = os.path.join(path_to_volumes2[ID_key2], data_settings['mri_filename'])

		subject = tio.Subject(mri1=tio.ScalarImage(image_path1),
					mri2=tio.ScalarImage(image_path2),
					label=label)
		train_subjects.append(subject)
	train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)

	val_subjects = []
	for pair, label in zip(X_val, y_val):
		ID_patient1 = pair[0].split('_')[0]
		ID_exam1 = pair[0].split('_')[1]
		ID_key1 = pair[0].split('_')[2]
		path_to_volumes1 = get_patient_seq_paths(data_settings['path_to_data'], ID_exam1, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient1)
		image_path1 = os.path.join(path_to_volumes1[ID_key1], data_settings['mri_filename'])

		ID_patient2 = pair[1].split('_')[0]
		ID_exam2 = pair[1].split('_')[1]
		ID_key2 = pair[1].split('_')[2]
		path_to_volumes2 = get_patient_seq_paths(data_settings['path_to_data'], ID_exam2, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient2)
		image_path2 = os.path.join(path_to_volumes2[ID_key2], data_settings['mri_filename'])

		subject = tio.Subject(mri1=tio.ScalarImage(image_path1),
				mri2=tio.ScalarImage(image_path2),
				label=label,
                                patient_id1=pair[0],
                                patient_id2=pair[1])
		val_subjects.append(subject)
	val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)

	train_sampler = CustomSampler(train_dataset, args.dataset_size-args.testset_size)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
	validation_loader = DataLoader(val_dataset, batch_size=args.batch_size)

	return train_loader, validation_loader

def get_loader_kidney_patient_disc(args, curriculum_step=0):
	data_settings = {
		'path_to_data': '/gpfs/workdir/mileckil/data/data_kidney_mri',
		'key_words_seqs': [['TUB', 'tub', 'WATER', 'AX', 'LAVA', ], ['WATER', 'AX', 'LAVA']],
		'mins_key_words': [4, 3],
		'exams': args.exams,
		'mri_filename': 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz',
		}

	dataset = []
	patients = next(os.walk(os.path.join(data_settings['path_to_data'])))[1]
	patients.remove('001-0048-O-E')
	for patient in patients:
		for exam in data_settings['exams']:
			if os.path.exists(os.path.join(data_settings['path_to_data'], patient, exam)):
				path_to_volumes = get_patient_seq_paths(data_settings['path_to_data'], exam, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=patient)
				if path_to_volumes:
					for key in path_to_volumes.keys():
						if os.path.exists(os.path.join(path_to_volumes[key], 'preprocessed')):
							dataset_id = patient + '_' + exam + '_' + key
							dataset.append(dataset_id)
	print('\nTotal dataset size: {}'.format(len(dataset)))
	X, y = [], []
	dataset2 = dataset.copy()
	for patient1 in dataset:
		dataset2.remove(patient1)
		for patient2 in dataset2:
			if (patient1.split('_')[0] == patient2.split('_')[0]) and (patient1.split('_')[1] == patient2.split('_')[1]):
				if curriculum_step==0:
					X.append([patient1, patient2])
					y.append(1)
				elif curriculum_step==1:
					pass
			elif (patient1.split('_')[0] == patient2.split('_')[0]) and (patient1.split('_')[1] != patient2.split('_')[1]):
				X.append([patient1, patient2])
				y.append(1)
			else:
				X.append([patient1, patient2])
				y.append(-1)
	X, y = np.array(X), np.array(y)
	print('\nTotal pairs size (positive examples): {}({})'.format(len(X), len(np.where(y==1)[0])))

	nb_pos_ex = len(np.where(y==1)[0])
	pos_ex = np.where(y==1)
	neg_ex = np.where(y==-1)
	neg_ex_to_keep = np.random.choice(neg_ex[0], size=10*args.dataset_size, replace=False)
	neg_ex_to_keep_val = np.random.choice(neg_ex_to_keep, size=3*args.testset_size//4, replace=False)
	neg_ex_to_keep_train = np.setdiff1d(neg_ex_to_keep, neg_ex_to_keep_val)
	pos_ex_to_keep = np.random.choice(pos_ex[0], size=args.dataset_size//4, replace=False)
	pos_ex_to_keep_val = np.random.choice(pos_ex_to_keep, size=args.testset_size//4, replace=False)

def get_loader_kidney(args, get_features=False):
	"""
	Building dataset and dataloader to get features or for downstream tasks.
	"""
	data_settings = {
		'path_to_data': '/gpfs/workdir/mileckil/data/data_kidney_mri',
		'path_to_targets': '/gpfs/workdir/mileckil/data/dataframes',
        	'key_words_seqs': [['TUB', 'tub', 'WATER', 'AX', 'LAVA', ]],
		'mins_key_words': [4],
		'exams': args.exams,
		'mri_filename': 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz',
		}

	test_ids = {'dfg_threshold_45': ['001-0064-Y-Y', '001-0012-B-C', '001-0056-B-A', '001-0066-G-G',
        '001-0024-C-P', '001-0047-T-M', '001-0107-Q-F', '001-0113-K-B',
        '001-0008-P-D', '001-0046-C-J', '001-0017-C-W', '001-0084-V-J',
        '001-0018-C-A', '001-0054-B-A', '001-0053-F-M', '001-0086-V-S',
        '001-0123-B-C', '001-0080-K-D', '001-0026-G-A', '001-0099-V-V'],
 'age_donneur': ['001-0022-H-F', '001-0055-B-Z', '001-0049-A-B', '001-0018-C-A',
        '001-0104-D-L', '001-0092-B-J', '001-0099-V-V', '001-0093-A-F',
        '001-0009-S-T', '001-0057-B-J', '001-0129-M-M', '001-0091-B-D',
        '001-0056-B-A', '001-0011-G-C', '001-0068-H-T', '001-0079-T-M',
        '001-0075-D-A', '001-0037-S-F', '001-0112-T-R', '001-0051-M-A'],
 'incomp_gref': ['001-0083-B-A', '001-0089-S-S', '001-0129-M-M', '001-0107-Q-F',
        '001-0086-V-S', '001-0109-C-B', '001-0079-T-M', '001-0043-F-M',
        '001-0046-C-J', '001-0056-B-A', '001-0125-A-J', '001-0112-T-R',
        '001-0092-B-J', '001-0096-S-M', '001-0119-R-A', '001-0061-P-A',
        '001-0060-L-J', '001-0093-A-F', '001-0131-S-S', '001-0071-B-M'],
 'nb_gref_prec': ['001-0098-R-N', '001-0039-C-R', '001-0037-S-F', '001-0045-D-D',
        '001-0096-S-M', '001-0034-W-N', '001-0065-Z-O', '001-0052-B-A',
        '001-0023-D-D', '001-0015-S-K', '001-0082-D-S', '001-0073-D-M',
        '001-0061-P-A', '001-0093-A-F', '001-0024-C-P', '001-0049-A-B',
        '001-0055-B-Z', '001-0086-V-S', '001-0025-B-B', '001-0008-P-D'],
 'dur_ischem_froide_m': ['001-0123-B-C', '001-0086-V-S', '001-0008-P-D', '001-0084-V-J',
        '001-0022-H-F', '001-0085-D-D', '001-0011-G-C', '001-0052-B-A',
        '001-0009-S-T', '001-0065-Z-O', '001-0060-L-J', '001-0096-S-M',
        '001-0102-M-L', '001-0121-B-C', '001-0113-K-B', '001-0072-E-J',
        '001-0075-D-A', '001-0109-C-B', '001-0091-B-D', '001-0103-D-A'],
 'Failure transplantation': ['001-0017-C-W', '001-0057-B-J', '001-0107-Q-F', '001-0114-M-D',
        '001-0121-B-C', '001-0015-S-K', '001-0049-A-B', '001-0125-A-J',
        '001-0062-R-R', '001-0026-G-A', '001-0012-B-C', '001-0010-T-R',
        '001-0064-Y-Y', '001-0099-V-V', '001-0083-B-A', '001-0085-D-D',
        '001-0008-P-D', '001-0109-C-B', '001-0046-C-J', '001-0047-T-M']} 
	test_ids = test_ids[args.target]
	test_ids = [patient+'_{}'.format(args.exams) for patient in test_ids]
        # Import target df
	if args.target.split('_')[0] == 'banff':
		target = args.target.split('_')[1]
		df_targets = pd.read_csv(os.path.join(data_settings['path_to_targets'], 'df_banff_targets.csv'), sep=';')
		if 'Unnamed: 0' in df_targets.columns:
			print('\nUnnamed 0 column removed')
			df_targets = df_targets.drop(columns='Unnamed: 0')
		select_cols = list(df_targets.columns)
		select_cols.remove('biopsie_comm')
		#df_targets = df_targets.dropna(axis=0, how='any', subset=select_cols)
		df_targets = df_targets.dropna(axis=0, how='any', subset=['biopsie_'+target])
		df_targets['exam'] = df_targets['visite'].apply(lambda x: x.split(' ')[1])
		df_targets = df_targets.drop(columns='visite')
		df_targets['patient'] = df_targets.apply(lambda x: x[0]+'_'+x[-1], axis=1)
		df_targets = df_targets.drop(columns='exam')
		labels = pd.Series((df_targets['biopsie_'+target].values>args.target_threshold).astype(int),index=df_targets['patient']).to_dict()

	else:
		target = args.target
		df_targets = pd.read_csv(os.path.join(data_settings['path_to_targets'], 'df_targets.csv'), sep=';')
		if 'Unnamed: 0' in df_targets.columns:
			print('\nUnnamed 0 column removed')
			df_targets = df_targets.drop(columns='Unnamed: 0')
		df_targets.dropna(axis=0, how='any', subset=[target])
		if args.exams is None:
			print("Please specify an followup exam !!")
		else:
			data_settings['exams'] = [args.exams]
			df_targets['patient'] = df_targets['patient']+'_'+args.exams
		labels = pd.Series((df_targets[target].values>args.target_threshold).astype(int),index=df_targets['patient']).to_dict()
	data_settings['labels'] = labels

	select_patients = []
	patients = next(os.walk(os.path.join(data_settings['path_to_data'])))[1]
	patients.remove('001-0048-O-E')
	for patient in patients:
		for exam in data_settings['exams']:
			if os.path.exists(os.path.join(data_settings['path_to_data'], patient, exam)):
				path_to_volumes = get_patient_seq_paths(data_settings['path_to_data'], exam, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=patient)
				if path_to_volumes:
					if len(list(path_to_volumes.keys()))==1:
						path_key = list(path_to_volumes.keys())[0]
						if os.path.exists(os.path.join(path_to_volumes[path_key], 'preprocessed')):
							patient_id = patient + '_' + exam
							select_patients.append(patient_id)
	select_patients = (set(data_settings['labels'].keys()) & set(select_patients))
	if not get_features:
        	select_patients = select_patients - set(test_ids)
	select_patients = list(select_patients)
	y = [data_settings['labels'][x] for x in select_patients]
	train_ids, eval_ids,_,_ = train_test_split(select_patients, y, test_size=args.evalset_size)
	print('\nTrain size (positive examples): {}({})'.format(len(train_ids), np.sum([data_settings['labels'][i] for i in train_ids])))
	print('Eval size (positive examples): {}({})\n'.format(len(eval_ids), np.sum([data_settings['labels'][i] for i in eval_ids])))
	# Verify that patient in test are not in train (other follow-up)
	patients_in_eval = np.unique([patient_id.split('_')[0] for patient_id in eval_ids])
	for train_id in train_ids:
		if train_id.split('_')[0] in patients_in_eval:
			train_ids.remove(train_id)
			test_ids.append(train_id)

	if args.augmentation==1:
		train_transform = tio.Compose([tio.RandomFlip(), 
					       tio.RandomAffine(p=0.5)])
		eval_transform = tio.Compose([])
	elif args.augmentation==2:
		train_transform = tio.Compose([tio.RandomFlip(),
					       tio.RandomAffine(p=0.5),
					       tio.RandomBlur((0,0.5)),
					       tio.RandomNoise(0, (0.05))])
		eval_transform = tio.Compose([])
	else:
		train_transform = tio.Compose([])
		eval_transform = tio.Compose([])
        
	if get_features:
		train_ids = select_patients
		eval_ids = []
		test_ids = []
	train_subjects = []
	for patient in train_ids:
		ID_patient = patient.split('_')[0]
		ID_exam = patient.split('_')[1]
		path_to_volumes = get_patient_seq_paths(data_settings['path_to_data'], ID_exam, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient)
		image_path = os.path.join(path_to_volumes[path_key], 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz')
		subject = tio.Subject(mri=tio.ScalarImage(image_path), label=data_settings['labels'][patient], patient_id=ID_patient) 
		train_subjects.append(subject)
	train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)

	eval_subjects = []
	for patient in eval_ids:
		ID_patient = patient.split('_')[0]
		ID_exam = patient.split('_')[1]
		path_to_volumes = get_patient_seq_paths(data_settings['path_to_data'], ID_exam, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient)
		image_path = os.path.join(path_to_volumes[path_key], 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz')
		subject = tio.Subject(mri=tio.ScalarImage(image_path), label=data_settings['labels'][patient], patient_id=patient)
		eval_subjects.append(subject)
	eval_dataset = [] if get_features else tio.SubjectsDataset(eval_subjects, transform=eval_transform)


	test_subjects = []
	for patient in test_ids:
		ID_patient = patient.split('_')[0]
		ID_exam = patient.split('_')[1]
		path_to_volumes = get_patient_seq_paths(data_settings['path_to_data'], ID_exam, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient)
		image_path = os.path.join(path_to_volumes[path_key], 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz')
		subject = tio.Subject(mri=tio.ScalarImage(image_path), label=data_settings['labels'][patient], patient_id=patient)
		test_subjects.append(subject)
	test_dataset = [] if get_features else tio.SubjectsDataset(test_subjects, transform=eval_transform)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
	eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
	print('\n len(train_loader)', len(train_loader))
	print('\n len(eval_loader)', len(eval_loader))
	print('\n len(test_loader)', len(test_loader))
	return train_loader, eval_loader, test_loader


