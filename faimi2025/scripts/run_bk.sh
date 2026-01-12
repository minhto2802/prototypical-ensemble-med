#!/bin/bash

#SBATCH -J d_t
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --time 1:00:00
##SBATCH --qos=m5
#SBATCH --gres=gpu:l40s:1
##SBATCH --gres=gpu:a40:1
#SBATCH --export=ALL
#SBATCH --output=logs/%x.%j.log
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90
#SBATCH --array=1-5
##SBATCH --exclude=gpu138,gpu180
#SBATCH --account=aip-medilab

# shellcheck disable=SC2155
export WANDB_RUN_ID="${SLURM_JOB_ID}_$(date +%Y%m%d-%H%M%S)"
export TQDM_MINITERS=10

seed=$SLURM_ARRAY_TASK_ID
if [ -z "$seed" ]; then
  seed=0 #
fi

python main.py \
  --seed $seed \
  --dpe.num_stages 50 \
  --dpe.epochs 20 \
  --dpe.lr 1e-3 \
  --dpe.cov_reg 5e4 \
  --dpe.batch_size_train 8 \
  --dpe.entropic 10 \
  --dpe.train_attr no \
  --dpe.emb_dim 2048 \
  --dpe.alpha 0.1 \
  --t.epochs 100 \
  --t.lr 1e-5 \
  --t.batch_size_train 8 \
  --t.d_model 128 \
  --t.ff_dim 512 \
  --dpe.trn_split 'va' \
  --t.trn_split 'va' \
  --data_dir "embeddings/bk/1830110" \
  --metadata_path "embeddings/bk/metadata_v1.csv" \
  --wdb_group bk_explore_v2.1 \
  "$@"
