#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16    # There are 40 CPU cores on Beluga GPU nodes
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --account=rrg-ebrahimi  # def-ibenayed
#SBATCH --output=%x-%j_%a.out

#SBATCH --mail-user=liubingyuan1988@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

# Load good version of python, and activate environment
module load python/3.8 cuda cudnn httpproxy
source ~/bing/bin/activate
pip install -e .

set -x

ME=FGADR_our-$(date +%m%d)
wandb offline
export WANDB_DIR=$SLURM_TMPDIR


# Prepare data and load them directly on the GPU node to make I/O faster
DATA_DIR=$SLURM_TMPDIR/data
mkdir $DATA_DIR
cp ~/scratch/Data/FGADR-Seg.tar.gz $DATA_DIR
tar xzf $DATA_DIR/FGADR-Seg.tar.gz -C $DATA_DIR

python tools/train_net.py segment --config-file ./configs/FGADR_bce-l1.yaml \
    --opts DATA.DATA_ROOT $DATA_DIR/FGADR-Seg/Seg-set

mkdir -p $HOME/wandb/$ME
mv $WANDB_DIR/wandb/offline* $HOME/wandb/$ME

