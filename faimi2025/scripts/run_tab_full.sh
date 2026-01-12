#!/bin/bash

#SBATCH -J d_t
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --time 1:00:00
#SBATCH --qos=m5
#SBATCH --gres=gpu:rtx6000:1
##SBATCH --gres=gpu:a40:1
#SBATCH --export=ALL
#SBATCH --output=logs/%x.%j.log
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90
##SBATCH --array=1-5
#SBATCH --exclude=gpu138,gpu180

# shellcheck disable=SC2155
export WANDB_RUN_ID="${SLURM_JOB_ID}_$(date +%Y%m%d-%H%M%S)"
export TQDM_MINITERS=10

python main.py \
  --seed 0 \
  --dpe.num_stages 50 \
  --dpe.epochs 40 \
  --dpe.lr 1e-4 \
  --dpe.cov_reg 5e4 \
  --dpe.batch_size_train 8 \
  --dpe.entropic 40 \
  --dpe.train_attr no \
  --dpe.emb_dim 192 \
  --t.epochs 30 \
  --t.lr 1e-5 \
  --t.batch_size_train 8 \
  --t.d_model 128 \
  --t.ff_dim 512 \
  --dpe.trn_split 'va' \
  --t.trn_split 'va' \
  --wdb_group tab_full_v2.2 \
  --data_dir embeddings/tab_full_v1 \
  --metadata_path embeddings/tab_full_v1/tabpfn_metadata_0_SURG.csv \
  "$@"

#  --multiclass True \
