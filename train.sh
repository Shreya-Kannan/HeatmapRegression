#!/bin/bash
#SBATCH --account=def-punithak
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --mail-user=skannan3@ualberta.ca
#SBATCH --mail-type=ALL

module load StdEnv/2020
module load python
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index torch numpy torchvision tensorboardX scipy pillow scikit-image scikit-learn matplotlib tqdm argparse

python trainer.py --epochs "$1" --mse_weight "$2" --dice_weight "$3" --kld_weight "$4" --bs "$5"