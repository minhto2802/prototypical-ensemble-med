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
  --dpe.num_stages 11 \
  --dpe.epochs 20 \
  --dpe.lr 1e-3 \
  --dpe.cov_reg 1e4 \
  --dpe.batch_size_train 8 \
  --dpe.entropic 50 \
  --dpe.train_attr no \
  --dpe.emb_dim 2048 \
  --t.epochs 200 \
  --t.lr 1e-4 \
  --t.batch_size_train 16 \
  --t.d_model 128 \
  --t.ff_dim 512 \
  --t.trn_split 'va' \
  --dpe.trn_split 'va' \
  --data_dir "embeddings/ham10k" \
  --metadata_path "embeddings/ham10k/metadata.csv" \
  --multiclass True \
  --wdb_group ham10k_multiclass_v1.2c \
  "$@"
