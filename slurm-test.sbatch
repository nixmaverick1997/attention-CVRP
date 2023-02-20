#!/bin/bash

#SBATCH --job-name=condaEnvSetup
#SBATCH --nodes=1
#SBATCH --partition=gpu_inter
#SBATCH --time=2:00:00

# Load the conda module
export PATH=/opt/conda/bin:$PATH
sudo apt-get install git
# Create conda environment 
# and prevent the use of system site-package
# that will interfer with the conda env
export PYTHONNOUSERSITE=1

# Create the environment using python 3.9
#conda create --name CVRP_env python=3.9 --force

# Activate the environment
source activate CVRP_env

# Install the required libraries from a requirement file, using pypi
# You can also trigger "conda install ..." commands
#jupyter notebook --no-browser
#python run.py --problem cvrp --n_epochs 1 --batch_size 128 --graph_size 50 --baseline rollout
#python generate_data.py --graph_size 100 --problem vrp --name test --seed 1234
#python eval.py data/vrp/vrp100_test_seed1234.pkl --model outputs/cvrp_100/run_20230213T025545 --decode_strategy greedy
#tensorboard --logdir=logs/cvrp_20/run_20230211T190453
#jupyter nbconvert plot_vrp.ipynb --to script
#python plot_vrp.py
git config --global user.email "nixmaverick97@outlook.com"
git config --global user.name "nixmaverick1997"
git add --all
git commit -m 'final commits- lkh, example solutions'