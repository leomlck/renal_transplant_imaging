import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse

import SimpleITK as sitk
import torchio as tio
import nibabel as nib

import sys
sys.path.insert(0, '../')
from  misc.data_import import get_patient_seq_paths

parser = argparse.ArgumentParser(description='MRI preprocessing')
parser.add_argument('--exam', metavar='EXAM', default='M12', type=str,
                    help='exam')
parser.add_argument("--img_size", default=[192, 144, 96], nargs='+', type=int,
                        help="Resolution size")
args = parser.parse_args()

path_to_data = '/kidney_seg/data/data_kidney_mri'
key_words_seqs = [['TUB', 'tub', 'WATER', 'AX', 'LAVA', ], ['WATER', 'AX', 'LAVA']]
mins_key_words = [4, 3]
path_keys = [key_words_seqs[0][0], key_words_seqs[1][0]]

select_patients = next(os.walk(path_to_data))[1]

for it_patient, select_patient in enumerate(select_patients):
	print('\nPatient id : {} ({}/{})'.format(select_patient, it_patient+1, len(select_patients)))

	if os.path.exists(os.path.join(path_to_data, select_patient, args.exam)):
		path_to_volumes = get_patient_seq_paths(path_to_data, args.exam, key_words_seqs, mins_key_words, select_patient=select_patient)
		if path_to_volumes:
			ref_img = nib.load(os.path.join(path_to_volumes['TUB'], 'cropped/mri_cropped.nii.gz'))
			for it_key, key in enumerate(path_to_volumes.keys()):
				print('\nKEY ID : {} ({}/{})'.format(key, it_key+1, len(path_to_volumes.keys())))
				
				if not os.path.exists(os.path.join(path_to_volumes[key], 'preprocessed')):
					os.mkdir(os.path.join(path_to_volumes[key], 'preprocessed'))
				# Recrop volume from original MRI and centered on already cropped ROI
				lx = max(0, (args.img_size[0]-ref_img.header['dim'][1])//2)
				ly = max(0, (args.img_size[1]-ref_img.header['dim'][2])//2)
				lz = max(0, (args.img_size[2]-ref_img.header['dim'][3])//2)	
				img = nib.load(os.path.join(path_to_volumes[key], 'mri.nii.gz'))
				x = abs(int((img.affine[0,3] - ref_img.affine[0,3]) / ref_img.affine[0,0]))
				y = abs(int((img.affine[1,3] - ref_img.affine[1,3]) / ref_img.affine[1,1]))
				z = abs(int((img.affine[2,3] - ref_img.affine[2,3]) / ref_img.affine[2,2]))
				cimg = img.slicer[max(0, x-lx):x+vit_size[0]-lx, max(0, y-ly):y+vit_size[1]-ly, max(0, z-lz):z+vit_size[2]-lz]
				nib.save(cimg, os.path.join(path_to_volumes[key], 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz'))
				
				cimg = sitk.ReadImage(os.path.join(path_to_volumes[key], 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz'))	
				statsFilter = sitk.StatisticsImageFilter()
				statsFilter.Execute(cimg)
				print(' '*8+'Original image:')
				print(' '*8+'Mean: {}, Std: {}'.format(statsFilter.GetMean(), statsFilter.GetSigma()))
				print(' '*8+'Min: {}, Max: {}'.format(statsFilter.GetMinimum(), statsFilter.GetMaximum()))

				# standard normization
				standardNormFilter = sitk.NormalizeImageFilter()
				ncimg = standardNormFilter.Execute(cimg)
				statsFilter.Execute(ncimg)
				print(' '*8+'After standard norm:')
				print(' '*8+'Mean: {}, Std: {}'.format(statsFilter.GetMean(), statsFilter.GetSigma()))
				print(' '*8+'Min: {}, Max: {}'.format(statsFilter.GetMinimum(), statsFilter.GetMaximum()))

				# clip to -5/5
				threshFilterLow = sitk.ThresholdImageFilter()
				threshFilterLow.SetUpper(statsFilter.GetMaximum()+1)
				threshFilterLow.SetLower(-5)
				threshFilterLow.SetOutsideValue(-5)
				ncimg = threshFilterLow.Execute(ncimg)
				threshFilterUp = sitk.ThresholdImageFilter()
				threshFilterUp.SetLower(-50)
				threshFilterUp.SetUpper(5)
				threshFilterUp.SetOutsideValue(5)
				ncimg = threshFilterUp.Execute(ncimg)
				statsFilter.Execute(ncimg)
				print(' '*8+'After clip to -5/5:')
				print(' '*8+'Mean: {}, Std: {}'.format(statsFilter.GetMean(), statsFilter.GetSigma()))
				print(' '*8+'Min: {}, Max: {}'.format(statsFilter.GetMinimum(), statsFilter.GetMaximum()))

				# rescale to [0,1]
				rescaleFilter = sitk.RescaleIntensityImageFilter()
				rescaleFilter.SetOutputMaximum(1)
				rescaleFilter.SetOutputMinimum(0)
				ncimg = rescaleFilter.Execute(ncimg)
				statsFilter.Execute(ncimg)
				print(' '*8+'After rescale to [0,1]:')
				print(' '*8+'Mean: {}, Std: {}'.format(statsFilter.GetMean(), statsFilter.GetSigma()))
				print(' '*8+'Min: {}, Max: {}'.format(statsFilter.GetMinimum(), statsFilter.GetMaximum()))
				sitk.WriteImage(ncimg, os.path.join(path_to_volumes[key], 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz'))
			
		
				# CropOrPad with tio
				ncimg = tio.ScalarImage(os.path.join(path_to_volumes[key], 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz'))
				tf = tio.CropOrPad(args.img_size)
				rncimg = tf(ncimg)
				rncimg.save(os.path.join(path_to_volumes[key], 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz'))
				

				# Save to float 32
				rncimg = sitk.ReadImage(os.path.join(path_to_volumes[key], 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz'), sitk.sitkFloat32)
				sitk.WriteImage(rncimg, os.path.join(path_to_volumes[key], 'preprocessed/mri_cropped_normalized_resized_vit.nii.gz'))


