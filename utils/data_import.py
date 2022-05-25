import os
import numpy as np

def get_patient_seq_paths(path_to_data, exam, key_words_seqs, mins_key_words, select_patient, dummy=False):
	"""
	Custom function to fetch paths to data according to subject id, key words for the sequences
	""" 
	patient = select_patient
	sequences = next(os.walk(os.path.join(path_to_data, patient, exam)))[1]
	path_to_volumes = {}
	seq_done = []
	for (key_words_seq, min_key_words) in zip(key_words_seqs, mins_key_words):
		for seq in sequences:
			if np.sum([key in seq for key in key_words_seq]) >= min_key_words:
				path_to_volume = os.path.join(path_to_data, patient, exam, seq)
				if dummy:
					path_to_volume = os.path.join(path_to_data, 'dummy_mri.nii.gz')
				if key_words_seq[0]=='WATER':
					count = seq.split('_')[0]
					path_to_volumes[key_words_seq[0]+'-{}'.format(count)] = path_to_volume
				else:
					path_to_volumes[key_words_seq[0]] = path_to_volume
					sequences.remove(seq)
	if len(path_to_volumes)<len(key_words_seqs):
		return False
	return path_to_volumes


