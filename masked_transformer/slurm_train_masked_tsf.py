import os
import io
import pandas as pd
import time

job_id = 'tsf_grid_0'
job_name = 'train_job.sh'

start_script = ('#!/bin/bash\n' +
                '#SBATCH --job-name=main_train_features_attention\n' +
                '#SBATCH --output=output/%x.o%j\n' +
                '#SBATCH --time=02:30:00\n' +
                '#SBATCH --ntasks=1\n' +
                '#SBATCH --cpus-per-task=1\n'
                '#SBATCH --mem=8GB\n' +
                '#SBATCH --gres=gpu:1\n' +
                '#SBATCH --partition=gpu\n' +
                '#SBATCH --export=NONE\n' +
                '\n' +
                'module load anaconda3/2021.05/gcc-9.2.0\n'+
                'module load cuda/10.2.89/intel-19.0.3.199\n'+
                'source activate pyenv\n')

command = 'python launch_grid_search.py --job_id {}'.format(job_id)

with open(job_name, 'w') as fh:
	fh.write(start_script)
	fh.write(command)
stdout = pd.read_csv(io.StringIO(os.popen("sbatch " + job_name).read()), delim_whitespace=True)
print(stdout)
os.remove(job_name)

