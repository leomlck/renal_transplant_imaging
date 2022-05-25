# Renal Transplantation & Medical Imaging project

## Contrastive Learning of Renal Transplant MRIs

Pretrain your ResNet Model locally
'''
python main_train_features.py
'''

Pretrain your ResNet Model sending a slurm job
'''
python slurm_train_features.py
'''

Infer & save the pretrained features to csv
'''
python get_features.py
'''
