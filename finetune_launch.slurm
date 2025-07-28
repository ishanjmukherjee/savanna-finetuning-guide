#!/bin/bash

#SBATCH --job-name=evo_finnatune
#SBATCH --nodes=1
#SBATCH --partition=gpu_batch
#SBATCH --output=log/%j_evo_finetune.log
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --time=8-00:00:00

# Tell PyTorch to build its allocator with expandable segments turned on
# Each of these flags is important and was determined using gnarly debugging
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True"
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=0
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export CUDNN_PATH="$CONDA_PREFIX"

# Change this based on how many GPUs you request
GPUS_PER_NODE=4

scontrol show hostname ${SLURM_JOB_NODELIST} > hostfile
sed -i "s/$/ slots=${GPUS_PER_NODE}/" hostfile

export SSH_OPTIONS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

MASTER_NODE=$(scontrol show hostname ${SLURM_JOB_NODELIST} | head -n 1)
CURR_NODE=$(hostname)
if [ "$CURR_NODE" = "$MASTER_NODE" ]; then
    while true
    do
        # cat hostfile

        python launch.py train.py -d configs data/evo2-ft-data.yml model/evo2-ft-model.yml

        sleep 900
    done
fi
