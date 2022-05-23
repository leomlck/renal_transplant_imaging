import os
import io
import pandas as pd
import time
import wandb

target = 'pred_M12'
job_description = 'timesformer_{}'.format(target) 
wandb_job_id = wandb.util.generate_id() 
job_name = '/gpfs/users/mileckil/kidney_workspace/project_kidney/timevit_workspace/src/loop_job_{}.sh'.format(job_description)

start_script = ('#!/bin/bash\n' +
                '#SBATCH --job-name=main_train_timesformer\n' +
                '#SBATCH --output=output/%x.o%j\n' +
                '#SBATCH --time=24:00:00\n' +
                '#SBATCH --ntasks=1\n' +
                '#SBATCH --cpus-per-task=1\n'
                '#SBATCH --mem=24GB\n' +
                '#SBATCH --gres=gpu:4\n' +
                '#SBATCH --partition=gpu\n' +
                '#SBATCH --export=NONE\n' +
                '\n' +
                'module load anaconda3/2021.05/gcc-9.2.0\n'+
                'module load cuda/11.2.0/intel-20.0.2\n'+
                'source activate pyenv2\n')
# from scratch

command = 'python train_timesformer.py --target {} --exams J15 J30 M3 M12 --mri_series TUB --val_size 10 --img_size 96 144 192 --batch_size 14 --eval_every 1 --learning_rate 1e-3 --num_epochs 200 --warmup_epochs 20 --augmentation 2 --vis_atn 0 --description {} --wandb_id {}'.format(target, job_description, wandb_job_id)

with open(job_name, 'w') as fh:
	fh.write(start_script)
	fh.write(command)
stdout = pd.read_csv(io.StringIO(os.popen("sbatch " + job_name).read()), delim_whitespace=True)
print(stdout)
os.remove(job_name)

JOBID = str(stdout.columns[-1])
#time.sleep(60)
#squeue = pd.read_csv(io.StringIO(os.popen('squeue').read()), delim_whitespace=True)
#job = squeue[squeue['USER']=='mileckil']
#JOBID = str(job['JOBID'].values[0])
print('New Job ID: {}'.format(JOBID))

command = command + ' --resume 1'
it=1
while True:
    it +=1
    time.sleep(60)
    sacct = pd.read_csv(io.StringIO(os.popen('sacct -j {}'.format(JOBID)).read()), delim_whitespace=True)
    job = sacct[sacct['JobID']==JOBID]
    print(sacct)
    print(job)
    if job['State'].values[0]=='TIMEOUT': 
        with open(job_name, 'w') as fh:
            fh.write(start_script)
            fh.write(command)
        stdout = pd.read_csv(io.StringIO(os.popen("sbatch " + job_name).read()), delim_whitespace=True)
        #os.system("sbatch " + job_name)
        os.remove(job_name)       
        JOBID = str(stdout.columns[-1]) 
        #time.sleep(60)
        #squeue = pd.read_csv(io.StringIO(os.popen('squeue').read()), delim_whitespace=True)
        #job = squeue[squeue['USER']=='mileckil']
        #JOBID = job['JOBID'].values[0]
        print('New Job ID: {}'.format(JOBID))
    elif job['State'].values[0]=='FAILED':
        print('Job {}, id {} failed'.format(it, JOBID))
        break
    elif job['State'].values[0]=='RUNNING':
        pass
    elif job['State'].values[0]=='PENDING':
        pass
    elif job['State'].values[0]=='COMPLETED':
        break



