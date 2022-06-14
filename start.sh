#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --time=20:40:00
#SBATCH --mem=30000
#SBATCH --job-name=gpu
#SBATCH --output=gpu.out
#SBATCH --error=gpu.err

module load anaconda/2021.11
source activate /p/tmp/bochow/ML_env/
module load cuda/10.2
export HDF5_USE_FILE_LOCKING='FALSE'

####RAW
#python train.py --root /p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/climate --batch_size 18 --n_threads 14 --max_iter 500000 --mask_root /p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/masks/sea_ice_missmask_full.h5 --save_dir /p/tmp/bochow/climatereconstructionAI-1.0.0/snapshots --log_dir /p/tmp/bochow/climatereconstructionAI-1.0.0/logs #--resume /home/kadow/pytorch/pytorch-hdf5-numpy/snapshots/20cr/ckpt/200000.pth
####FINE
#python train.py --root /p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/climate --mask_root /p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/masks/sea_ice_missmask_full.h5 --finetune --resume /p/tmp/bochow/climatereconstructionAI-1.0.0/snapshots/ckpt/500000.pth --batch_size 18 --n_threads 8 --max_iter 1000000 --save_dir /p/tmp/bochow/climatereconstructionAI-1.0.0/snapshots/finetune  --log_dir /p/tmp/bochow/climatereconstructionAI-1.0.0/logs
####TEST
python test.py --root /p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/climate --mask_root /p/tmp/bochow/climatereconstructionAI-1.0.0/FREVA-CLINT-climatereconstructionAI-2fc5e62/masks/sea_ice_missmask_full.h5 --snapshot /p/tmp/bochow/climatereconstructionAI-1.0.0/snapshots/finetune/ckpt/1000000.pth
