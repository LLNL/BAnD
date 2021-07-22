# Brain Attend and Decode (BAnD)
- Paper: Attend and Decode: 4D fMRI Task State Decoding Using Attention Models, https://arxiv.org/abs/2004.05234

## Structure:
- band/: src code for BAnD related models and utilities
- main.py: train BAnD model without distributed mode
- main_dist.py: train BAnD model with distributed mode
- main_dist_finetune.py: finetune pre-trained BAnD to target dataset
- start_dist.sh: helper bash script to start training models with provided hyperparameters and directories
- constants.py: dataset and output directories constants

## Pre-trained weights:
- Weights of BAnD model pre-trained on a large dataset of fMRI data: Human Connectome Project.
- We're releasing the top 3 weights in terms of validation loss. We recommend using the best weight but one can potentially average the 3 sets of weights to get some ensemble effects.
- See Releases for weights: https://github.com/LLNL/BAnD/releases
