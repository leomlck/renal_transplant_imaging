# Renal Transplantation & Medical Imaging project

## Contrastive Learning of Renal Transplant MRIs

<p align="center">
  <img src="figures/overview.png" width="700">
</p>

### Usage

Pretrain your ResNet Model locally
```
python main_train_features.py
```

Pretrain your ResNet Model sending a slurm job
```
python slurm_train_features.py
```

Infer & save the pretrained features to csv
```
python get_features.py
```

### Dummy dataset
As the dataset for this work is not publicly available, I built a dummy mri dataset path tree similar to our dataset so that the code can be ran on it, when argument ```dummy=True``` in ```get_patient_seq_paths``` function
```bash
├── data
│   ├── dummy_dataframes
│   │   ├── df_targets.csv
│   ├── dummy_mri_dataset (contains patients)
│   │   ├── dummy_mri.nii.gz
│   │   ├── 001-0001-A-A (contains exams)
│   │   │   ├── D15 (contains MRI sequences)
│   │   │   │   ├── 1_WATER_AX_LAVA-Flex_ss_IV
│   │   │   │   ├── 2_WATER_AX_LAVA-Flex_ART
│   │   │   │   ├── 3_WATER_AX_LAVA-Flex_tub
│   │   │   ├── D30
│   │   │   ├── M3
│   │   │   ├── M12
│   │   ├── 001-0002-B-B
│   │   ├── ...
└── ...
```

### Requirements
See conda_environment.yml file
Or
```
conda env create -n ENVNAME --file conda_environment.yml
```

### Visualization of features
<p align="center">
  <img src="figures/visualization.png" width="700">
</p>

### Reference
```
@inproceedings{milecki2022constrative,
  title={Constrative Learning for Kidney Transplant Analysis using {MRI} data and Deep Convolutional Networks},
  author={Leo Milecki and Vicky Kalogeiton and Sylvain Bodard and Dany Anglicheau and Jean-Michel Correas and Marc-Olivier Timsit and Maria Vakalopoulou},
  booktitle={Medical Imaging with Deep Learning},
  year={2022},
  url={https://openreview.net/forum?id=fLUyt7-mWwI}
}
```
