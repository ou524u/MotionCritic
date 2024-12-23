# Ideal Directory Structure Documentation

This document outlines the ideal directory structure for the project. Follow the steps detailed in the README.md file to create and organize the directories and files as described below.

## Directory Structure Overview

The project is structured as follows:

```bash
$ tree -L 2
.
├── docs
│   ├── annotation.md
│   ├── annotation.png
│   ├── dataset.md
│   ├── finetuning.md
│   ├── metadata.json
│   └── motion.md
├── environment.yml
├── MDMCritic # Finetuning MDM with MotionCritic
│   ├── critic
│   ├── data_loaders
│   ├── diffusion
│   ├── eval
│   ├── model
│   ├── render
│   ├── sample
│   ├── train
│   ├── utils
│   └── visualize
├── MotionCritic # Core MotionCritic components
│   ├── data # Full motion data (optional)
│   ├── datasets # Processed motion data (e.g., mlists.zip)
│   ├── lib
│   ├── marked # Motion annotations
│   ├── metric
│   ├── motion_pair_dataset.py
│   ├── parsedata.py
│   ├── render
│   ├── train.py
│   └── visexample.py
├── prepare # Bash scripts for dataset preparation
│   ├── prepare_dataset.sh
│   ├── prepare_demo.sh
│   ├── prepare_fullannotation.sh
│   ├── prepare_fullmotion.sh
│   ├── prepare_MDM_dataset.sh
│   ├── prepare_MDM_pretrained.sh
│   ├── prepare_pretrained.sh
│   └── prepare_smpl.sh
├── README.md
└── requirements.txt
```
