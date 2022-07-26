import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

def get_radiomics_features(args, path_to_features, exams):
	"""
	Load radiomics features as panda dataframe per exam in a dictionnary, normalize if asked.
	""" 
	if args.normalize:
		scaler = make_pipeline(StandardScaler(), MinMaxScaler())
	dict_features = {}
	for exam in exams:
		path = os.path.join(path_to_features, 'radiomics_features_{}.csv'.format(exam))
		df = pd.read_csv(path)
		df['aug'] = 0
		if args.normalize:
			df_n = df.drop(columns=['patient', 'aug'])
			scaler.fit(df_n)
			scaled_data = scaler.transform(df_n)
			df_n = pd.DataFrame(scaled_data, columns=df_n.columns, index=df_n.index)
			df = pd.concat([df_n, df[['patient', 'aug']]], axis=1)
			dict_features[exam] = df
	return dict_features

def get_contrastive_features(args, path_to_features, exams):
	"""
	Load imaging features pretrained using our contrastive scheme as panda dataframe per exam in a dictionnary, normalize if asked
	"""
	if args.normalize:
		scaler = make_pipeline(StandardScaler(), MinMaxScaler())
	dict_features = {}
	for exam in exams:
		path = os.path.join(path_to_features, 'contrastive_features_{}_{}.csv'.format(exam, args.pretraining))
		df = pd.read_csv(path)
		df['aug'] = 0
		if args.augmentation:  
			path_to_features_aug = os.path.join(path_to_features, 'contrastive_features_{}_{}_aug.csv'.format(exam, args.pretraining))
			df_features_aug = pd.read_csv(path_to_features_aug)
			df_features_aug['aug'] = 1
			df = pd.concat([df, df_features_aug], axis=0)
		if args.normalize:
			df_n = df.drop(columns=['patient', 'aug'])
			scaler.fit(df_n)
			scaled_data = scaler.transform(df_n)
			df_n = pd.DataFrame(scaled_data, columns=df_n.columns, index=df_n.index)
			df = pd.concat([df_n, df[['patient', 'aug']]], axis=1)
			dict_features[exam] = df
	return dict_features

def prepare_biological_data(args, path_to_biological_data):
	"""
	Load and preprocess biological data (serum creatinine) to panda dataframe.
	"""
	df_biological = pd.read_csv(os.path.join(path_to_biological_data, 'df_biological_data.csv'))

	clip_value = df_biological['sCreatinine'].quantile(0.95)
	df_biological[['sCreatinine']] = df_biological[['sCreatinine']].clip(upper=clip_value)
	sCreat_scaler = make_pipeline(StandardScaler(), MinMaxScaler())
	df_n = df_biological[['sCreatinine']]
	sCreat_scaler.fit(df_n)
	scaled_data = sCreat_scaler.transform(df_n)
	df_n = pd.DataFrame(scaled_data, columns=df_n.columns, index=df_n.index)
	df_biological['sCreatinine'] = df_n
	return df_biological, sCreat_scaler, clip_value

def get_sCreat_data_per_followup(df_biological, exams, label_dates):
	"""
	Return sCreat values available in between each exam for each patient.
	"""
	for label_date in list(label_dates.keys()):
		df_biological[label_date] = label_dates[label_date]
	sCreat_data = {}
	for patient in np.unique(df_biological['patient']):
		sCreat_data[patient] = {}
		last_exam_date = -2000
		for exam in list(exams.keys()) + list(label_dates.keys()):
			df = df_biological[(df_biological['patient']==patient) & (last_exam_date<df_biological['days_after_transplant']) & (df_biological['days_after_transplant']<=df_biological[exam])]
			sCreat_data[patient][exam] = np.stack((df['days_after_transplant'].values, df['sCreatinine'].values))
			last_exam_date = df_biological[exam]
		df = df_biological[(df_biological['patient']==patient) & (300<=df_biological['days_after_transplant'])]
		sCreat_data[patient]['pred'] = np.stack((df['days_after_transplant'].values, df['sCreatinine'].values))
	return sCreat_data

def get_sCreat_features(sCreat_data, exams, label_dates, seq_len):
	"""
	Get sCreat features (mean, std, max, min, median, number of points) between each exam.
	Also build sCreat_masks that tag if no points are available for an interval.
	"""
	sCreat_data_stats = {}
	sCreat_masks = {}
	for patient in sCreat_data.keys():
		sCreat_data_stats[patient] = {}
		sCreat_masks[patient] = {'src': np.zeros(seq_len+1), 'tgt': np.zeros(seq_len)}
		for it_exam, exam in enumerate(list(exams.keys())[1:]):
			if sCreat_data[patient][exam].shape[1]==0:
				sCreat_data_stats[patient][exam] = -np.zeros(6)
				sCreat_masks[patient]['src'][it_exam+1] = 1
			else:
				sCreat_data_stats[patient][exam] = np.array([sCreat_data[patient][exam].shape[1],
							    np.mean(sCreat_data[patient][exam][1]),
							    np.median(sCreat_data[patient][exam][1]),
							    np.std(sCreat_data[patient][exam][1]),
							    np.min(sCreat_data[patient][exam][1]),
							    np.max(sCreat_data[patient][exam][1])])
		for it_exam, exam in enumerate(label_dates.keys()):
			if sCreat_data[patient][exam].shape[1]==0:
				sCreat_data_stats[patient][exam] = -np.zeros(5)
				sCreat_masks[patient]['tgt'][it_exam] = 1
			else:
				sCreat_data_stats[patient][exam] = np.array([np.mean(sCreat_data[patient][exam][1]),
							    np.median(sCreat_data[patient][exam][1]),
							    np.std(sCreat_data[patient][exam][1]),
							    np.min(sCreat_data[patient][exam][1]),
							    np.max(sCreat_data[patient][exam][1])])
	return sCreat_data_stats, sCreat_masks

def get_features_masks(dict_features, sCreat_data_stats, exams, seq_len):
	"""
	Build imaging features masks that tag which exams are available for each patient.
	"""
	feat_masks = {}
	for patient in sCreat_data_stats.keys():
		feat_masks[patient] = np.zeros(seq_len+1)
		for i_exam, exam in enumerate(list(exams.keys())[1:]):
			if patient not in list(dict_features[exam]['patient']):
				feat_masks[patient][i_exam+1] = 1
	return feat_masks


def get_sCreat_labels(sCreat_data, label_dates, preds_dist_thresold=61):
	"""
	Get the sCreat label as the mean value arround label_dates with a interval of +/- preds_dist_thresold days.
	"""
	sCreat_labels = {}
	for patient in sCreat_data.keys():
		sCreat_labels[patient] = {}
		for label_date in label_dates.keys():
			if sCreat_data[patient]['pred'].shape[1]==0:
				sCreat_labels.pop(patient, None)
			else:
				dist_to_pred = abs(sCreat_data[patient]['pred'][0]-label_dates[label_date])
				idx = np.where(dist_to_pred<preds_dist_thresold)
				if len(idx[0]) != 0:
					sCreat_labels[patient][label_date] = np.mean(sCreat_data[patient]['pred'][1][idx])
				else:
					sCreat_labels[patient][label_date] = np.nan
	return sCreat_labels

def value_to_class(value, thresholds):
	"""
	Transform labels to class according to some predefined thresholds.
	"""
	lst = list(thresholds).copy()
	lst.append(value)
	lst.sort()
	return lst.index(value)

def sCreat_labels_to_classes(sCreat_labels, sCreat_scaler, thresholds=[110]):
	"""
	Transform labels to class according to some predefined thresholds.
	"""
	n_labels = len(thresholds)
	thresholds = sCreat_scaler.transform(np.array(thresholds).reshape(-1, 1))
	sCreat_class_labels = {}
	for patient in sCreat_labels.keys():
		sCreat_class_labels[patient] = {}
		for label_date in sCreat_labels[patient].keys():
			if ~np.isnan(sCreat_labels[patient][label_date]):
				sCreat_class_labels[patient][label_date] = value_to_class(sCreat_labels[patient][label_date], thresholds)
	return sCreat_class_labels, n_labels

def get_available_patients_sCreat(args, sCreat_data_stats, sCreat_labels, sCreat_masks, min_seq_len=3):
	"""
	Get list of patients that has maximum min_seq_len missing exams features for Screat.
	"""
        input_size = 6
        sCreat_patients = []
        for patient in sCreat_data_stats.keys():
                if np.sum(sCreat_masks[patient]['src'])<=min_seq_len:
                        sCreat_patients.append(patient)
        target_patients = set(np.array(list(sCreat_labels.keys()))[(~np.isnan(np.stack([sCreat_labels[patient][args.target] for patient in sCreat_labels.keys()])))])
        patients = list(set(sCreat_patients) & target_patients)
        print('Number of patients: ', len(patients))
        return patients, input_size

def get_available_patients_features(args, dict_features, sCreat_labels, exams):
	"""
	Get list of patients that has maximum min_seq_len missing exams features for contrastive/radiomics features.
	"""
	if args.input == 'radiomics':
		input_size = 107 
		radiomics_patients = set()
		for exam in list(exams.keys())[1:]:
			radiomics_patients = radiomics_patients | set(dict_features[exam]['patient'])
		target_patients = set(np.array(list(sCreat_labels.keys()))[(~np.isnan(np.stack([sCreat_labels[patient][args.target] for patient in sCreat_labels.keys()])))])
		patients = list(radiomics_patients & target_patients)
		print('Number of patients: ', len(patients))
	elif args.input == 'img':
		input_size = 512
		mri_patients = set()
		for exam in list(exams.keys())[1:]:
			mri_patients = mri_patients | set(dict_features[exam]['patient'])
		target_patients = set(np.array(list(sCreat_labels.keys()))[(~np.isnan(np.stack([sCreat_labels[patient][args.target] for patient in sCreat_labels.keys()])))])
		patients = list(mri_patients & target_patients)
		print('Number of patients: ', len(patients))
	return patients, input_size


