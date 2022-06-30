import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

import os
import numpy as np
import pandas as pd
import torchio as tio
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '../')
from  utils.misc import get_patient_seq_paths

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
	Building dataset and dataloader for the weakly-supervised contrastive learning setup (using clinical variable information).
	"""
	data_settings = {
		'path_to_data': '../data/dummy_mri_dataset',
		'path_to_targets': '../data/dummy_dataframes',
		'key_words_seqs': [['TUB', 'tub', 'WATER', 'AX', 'LAVA', ], ['WATER', 'AX', 'LAVA']],
		'mins_key_words': [4, 3],
		'exams': args.exams,
		'mri_filename': 'dummy_mri.nii.gz',
		}

	df_targets = pd.read_csv(os.path.join(data_settings['path_to_targets'], 'df_targets.csv'))
	if args.target == 'GFR':
		df_targets = df_targets[['patient']+[args.target+' {}'.format(exam) for exam in args.exams]]
	else:
		df_targets = df_targets[['patient', args.target]]

	# Fetch all the available exams with the corresponding clinical information value	
	dataset = []
	patients = next(os.walk(os.path.join(data_settings['path_to_data'])))[1]
	for patient in patients:
		for exam in data_settings['exams']:
			if os.path.exists(os.path.join(data_settings['path_to_data'], patient, exam)):
				path_to_volumes = get_patient_seq_paths(data_settings['path_to_data'], exam, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=patient, dummy=True)
				if path_to_volumes:
					try:
						if args.target == 'GFR':
							val = df_targets.loc[df_targets['patient']==patient][args.target+' {}'.format(exam)].values[0]
						else:
							val = df_targets.loc[df_targets['patient']==patient][args.target].values[0]
					except (KeyError, IndexError):
						val = np.nan
					if not np.isnan(val):
						for key in path_to_volumes.keys():
							if os.path.exists(path_to_volumes[key]):
								sample_id = patient + '_' + exam + '_' + key
								dataset.append((sample_id, val))
	# Build a dataset of pairs, set label y of the pair according to the difference en clinical information value and a fixed threshold
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
	neg_ex_to_keep_val = np.random.choice(neg_ex_to_keep, size=3*args.valset_size//4, replace=False)
	neg_ex_to_keep_train = np.setdiff1d(neg_ex_to_keep, neg_ex_to_keep_val)
	pos_ex_to_keep = np.random.choice(pos_ex[0], size=args.dataset_size//4, replace=False)
	pos_ex_to_keep_val = np.random.choice(pos_ex_to_keep, size=args.valset_size//4, replace=False)
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
		path_to_volumes1 = get_patient_seq_paths(data_settings['path_to_data'], ID_exam1, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient1, dummy=True)
		image_path1 = os.path.join(path_to_volumes1[ID_key1], data_settings['mri_filename'])

		ID_patient2 = pair[1].split('_')[0]
		ID_exam2 = pair[1].split('_')[1]
		ID_key2 = pair[1].split('_')[2]
		path_to_volumes2 = get_patient_seq_paths(data_settings['path_to_data'], ID_exam2, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient2, dummy=True)
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
		path_to_volumes1 = get_patient_seq_paths(data_settings['path_to_data'], ID_exam1, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient1, dummy=True)
		image_path1 = os.path.join(path_to_volumes1[ID_key1], data_settings['mri_filename'])

		ID_patient2 = pair[1].split('_')[0]
		ID_exam2 = pair[1].split('_')[1]
		ID_key2 = pair[1].split('_')[2]
		path_to_volumes2 = get_patient_seq_paths(data_settings['path_to_data'], ID_exam2, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient2, dummy=True)
		image_path2 = os.path.join(path_to_volumes2[ID_key2], data_settings['mri_filename'])

		subject = tio.Subject(mri1=tio.ScalarImage(image_path1),
				mri2=tio.ScalarImage(image_path2),
				label=label,
                                patient_id1=pair[0],
                                patient_id2=pair[1])
		val_subjects.append(subject)
	val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)

	train_sampler = CustomSampler(train_dataset, args.dataset_size-args.valset_size)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
	validation_loader = DataLoader(val_dataset, batch_size=args.batch_size)

	return train_loader, validation_loader

def get_loader_kidney_patient_disc(args, curriculum_step=0):
	"""
	Building dataset and dataloader for the self-supervised contrastive learning setup (patient level).
	"""
	data_settings = {
		'path_to_data': '../data/dummy_mri_dataset',
		'key_words_seqs': [['TUB', 'tub', 'WATER', 'AX', 'LAVA', ], ['WATER', 'AX', 'LAVA']],
		'mins_key_words': [4, 3],
		'exams': args.exams,
		'mri_filename': 'dummy_mri.nii.gz',
		}
	
	# Fetch all the available exams with the corresponding clinical information value
	dataset = []
	patients = next(os.walk(os.path.join(data_settings['path_to_data'])))[1]
	for patient in patients:
		for exam in data_settings['exams']:
			if os.path.exists(os.path.join(data_settings['path_to_data'], patient, exam)):
				path_to_volumes = get_patient_seq_paths(data_settings['path_to_data'], exam, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=patient, dummy=True)
				if path_to_volumes:
					for key in path_to_volumes.keys():
						sample_id = patient + '_' + exam + '_' + key
						dataset.append(sample_id)
	# Build a dataset of pairs, set label y of the pair according to the patient / exam ids
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

	# Fix the dataset size, the positive sample pairs (25%) and make a pool of negative samples
	nb_pos_ex = len(np.where(y==1)[0])
	pos_ex = np.where(y==1)
	neg_ex = np.where(y==-1)
	neg_ex_to_keep = np.random.choice(neg_ex[0], size=10*args.dataset_size, replace=False)
	neg_ex_to_keep_val = np.random.choice(neg_ex_to_keep, size=3*args.valset_size//4, replace=False)
	neg_ex_to_keep_train = np.setdiff1d(neg_ex_to_keep, neg_ex_to_keep_val)
	pos_ex_to_keep = np.random.choice(pos_ex[0], size=args.dataset_size//4, replace=False)
	pos_ex_to_keep_val = np.random.choice(pos_ex_to_keep, size=args.valset_size//4, replace=False)
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
		path_to_volumes1 = get_patient_seq_paths(data_settings['path_to_data'], ID_exam1, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient1, dummy=True)
		image_path1 = os.path.join(path_to_volumes1[ID_key1], data_settings['mri_filename'])

		ID_patient2 = pair[1].split('_')[0]
		ID_exam2 = pair[1].split('_')[1]
		ID_key2 = pair[1].split('_')[2]
		path_to_volumes2 = get_patient_seq_paths(data_settings['path_to_data'], ID_exam2, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient2, dummy=True)
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
		path_to_volumes1 = get_patient_seq_paths(data_settings['path_to_data'], ID_exam1, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient1, dummy=True)
		image_path1 = os.path.join(path_to_volumes1[ID_key1], data_settings['mri_filename'])

		ID_patient2 = pair[1].split('_')[0]
		ID_exam2 = pair[1].split('_')[1]
		ID_key2 = pair[1].split('_')[2]
		path_to_volumes2 = get_patient_seq_paths(data_settings['path_to_data'], ID_exam2, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient2, dummy=True)
		image_path2 = os.path.join(path_to_volumes2[ID_key2], data_settings['mri_filename'])

		subject = tio.Subject(mri1=tio.ScalarImage(image_path1),
				mri2=tio.ScalarImage(image_path2),
				label=label,
                                patient_id1=pair[0],
                                patient_id2=pair[1])
		val_subjects.append(subject)
	val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)
	
	train_sampler = CustomSampler(train_dataset, args.dataset_size-args.valset_size)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
	validation_loader = DataLoader(val_dataset, batch_size=args.batch_size)

	return train_loader, validation_loader


def get_loader_kidney(args):
	"""
	Building dataset and dataloader to get features.
	"""
	data_settings = {
		'path_to_data': '../data/dummy_mri_dataset',
		'path_to_targets': '../data/dummy_dataframes',
        	'key_words_seqs': [['TUB', 'tub', 'WATER', 'AX', 'LAVA', ]],
		'mins_key_words': [4],
		'exams': [args.exams],
		'mri_filename': 'dummy_mri.nii.gz',
		}

	select_patients = []
	patients = next(os.walk(os.path.join(data_settings['path_to_data'])))[1]
	for patient in patients:
		for exam in data_settings['exams']:
			if os.path.exists(os.path.join(data_settings['path_to_data'], patient, exam)):
				path_to_volumes = get_patient_seq_paths(data_settings['path_to_data'], exam, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=patient, dummy=True)
				if path_to_volumes:
					if len(list(path_to_volumes.keys()))==1:
						path_key = list(path_to_volumes.keys())[0]
						patient_id = patient + '_' + exam
						select_patients.append(patient_id)
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
        
	train_ids = select_patients
	train_subjects = []
	for patient in train_ids:
		ID_patient = patient.split('_')[0]
		ID_exam = patient.split('_')[1]
		path_to_volumes = get_patient_seq_paths(data_settings['path_to_data'], ID_exam, data_settings['key_words_seqs'], data_settings['mins_key_words'], select_patient=ID_patient, dummy=True)
		image_path = os.path.join(path_to_volumes[path_key], data_settings['mri_filename'])
		subject = tio.Subject(mri=tio.ScalarImage(image_path), patient_id=ID_patient) 
		train_subjects.append(subject)
	train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

	return train_loader


