import os
import io
import pandas as pd
import time
import wandb

architecture = 'resnet50'
target = 'age_donor'
job_description = '{}_{}_cosloss_augmentation2_dropout_margin'.format(architecture, target) 
wandb_job_id = wandb.util.generate_id() 
job_name = '/gpfs/users/mileckil/kidney_workspace/project_kidney/radiomics_workspace/src/loop_job_{}.sh'.format(job_description)

start_script = ('#!/bin/bash\n' +
                '#SBATCH --job-name=main_train_features_cnn_resume\n' +
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
# from scratch

command = 'python train_features_cnn_resume_pairs.py --target {} --exams J15 J30 M3 M12 --dataset_size 10500 --testset_size 500 --architecture {} --img_size 96 144 192 --features_head mlp --feat_dim 256 --batch_size 30 --eval_every 1 --learning_rate 1e-2 --num_epochs 20 40 60 --warmup_epochs 5 --normalize_feat 1 --augmentation 2 --dropout 0.1 --curriculum 5 30 10 20 10 15 --loss_margin 0.5 --description {} --wandb_id {}'.format(target, architecture, job_description, wandb_job_id)


# from imagenet
'''
command = 'python train_features_cnn_resume_pairs.py --target {} --exams J15 J30 M3 --dataset_size 5000 --testset_size 500 --architecture {} --pretrained_dir /gpfs/workdir/mileckil/data/pretrained_models/2d_resnet/r2d18_imagenet.bin --img_size 96 144 192 --features_head mlp --feat_dim 256 --train_batch_size 24 --eval_batch_size 24 --eval_every 1 --learning_rate 1e-3 --num_epochs 30 60 --warmup_epochs 5 --normalize_feat 1 --augmentation 2 --dropout 0.1 --curriculum 15 35 10 20 --loss_margin 0.5 --description {} --wandb_id {}'.format(target, architecture, job_description, wandb_job_id)
'''

# recover
'''
command = 'python train_features_cnn_resume_pairs.py --target {} --exams J15 J30 M3 --dataset_size 5000 --testset_size 500 --architecture {} --img_size 96 144 192 --features_head mlp --feat_dim 256 --train_batch_size 24 --eval_batch_size 24 --eval_every 1 --learning_rate 1e-2 --num_epochs 20 40 60 --warmup_epochs 5 --normalize_feat 1 --augmentation 2 --dropout 0.1 --curriculum 1 3 1 2 1 1 --loss_margin 0.5 --description {} --wandb_id {} --resume 1'.format(target, architecture, job_description, wandb_job_id)
'''

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



