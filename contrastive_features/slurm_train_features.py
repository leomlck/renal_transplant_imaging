import os
import io
import pandas as pd
import time
import wandb

architecture = 'resnet18'
target = 'age_donor'
job_description = '{}_{}_contrastive_cosloss'.format(architecture, target) 

wandb_job_id = wandb.util.generate_id() 
job_name = 'loop_job_{}.sh'.format(job_description)
start_script = ('#!/bin/bash\n' +
                '#SBATCH --job-name=main_train_features\n' +
                '#SBATCH --output=output/%x.o%j\n' +
                '#SBATCH --time=24:00:00\n' +
                '#SBATCH --ntasks=1\n' +
                '#SBATCH --cpus-per-task=4\n'
                '#SBATCH --mem=12GB\n' +
                '#SBATCH --gres=gpu:4\n' +
                '#SBATCH --partition=gpu\n' +
                '#SBATCH --export=NONE\n' +
                '\n' +
                'module load anaconda3/2021.05/gcc-9.2.0\n'+
                'module load cuda/10.2.89/intel-19.0.3.199\n'+
                'source activate pyenv\n')

# from scratch python training
command = 'python main_train_features.py --target {} --exams J15 J30 M3 M12 --dataset_size 10500 --testset_size 500 --architecture {} --img_size 96 144 192 --features_head mlp --feat_dim 256 --batch_size 30 --eval_every 1 --learning_rate 1e-2 --num_epochs 20 40 60 --warmup_epochs 5 --normalize_feat 1 --augmentation 2 --dropout 0.1 --curriculum 5 30 10 20 10 15 --loss_margin 0.5 --description {} --wandb_id {}'.format(target, architecture, job_description, wandb_job_id)


# send slurm job
with open(job_name, 'w') as fh:
	fh.write(start_script)
	fh.write(command)
stdout = pd.read_csv(io.StringIO(os.popen("sbatch " + job_name).read()), delim_whitespace=True)
print(stdout)
os.remove(job_name)

JOBID = str(stdout.columns[-1])
print('New Job ID: {}'.format(JOBID))

# watch job state, if ended due to time limit, resume training
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
        os.remove(job_name)       
        JOBID = str(stdout.columns[-1]) 
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



