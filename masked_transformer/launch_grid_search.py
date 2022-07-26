import os
import numpy as np
import argparse
import subprocess
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(description='Launch kidney features attention')
parser.add_argument("--job_id", type=str)
args = parser.parse_args()

parameters = {'target': ['pred_M12'],
 'model': ['tsf'],
 'input': ['img'],
 'pretraining': ['dummy'],
 'normalize': [1],
 'batch_size': [32],
 'n_heads': [2], #[1, 2, 4, 8], #[int(x) for x in np.linspace(1,4,4)],
 'n_layers': [2], #[1, 2, 3, 4], #[int(x) for x in np.linspace(1,4,4)],
 'ffwd_dim': [768], #[64, 128, 256, 512, 768],
 'dropout': [0.1],
 'lr': [1e-3],
 'epochs': [30],
 'warmup_epochs': [5],
 'n_splits': [10],
 'augmentation': [0],
 'weight_decay': [0.],
}

for i, params in enumerate(list(ParameterGrid(parameters))):
	print('Sending job params {}/{}'.format(i+1, len(list(ParameterGrid(parameters)))))
	job_description = 'encoder_{}_{}_{}_params_{}'.format(params['model'], params['input'], args.job_id, i)+'_{}'.format(params['pretraining'])
	params['description'] = job_description
	params_list = ['--{} {}'.format(param, params[param]) for param in params.keys()]
	command = 'python main_train_ked_tsf.py ' + ' '.join(params_list)
	subprocess.Popen(command, shell=True).wait()

