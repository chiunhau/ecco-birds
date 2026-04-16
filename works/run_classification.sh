#!/bin/bash
#SBATCH --account=project_2017429
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:v100:4

module load pytorch/2.5

export HF_HOME=/scratch/project_2017429/chiunhau/birds/hf-cache

MODEL="Qwen/Qwen2.5-32B-Instruct"

LOG=/scratch/project_2017429/chiunhau/birds/works/vllm-${SLURM_JOB_ID}.log

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --tensor-parallel-size 4 \
  --dtype half \
  --max-model-len 2048 \
  > $LOG &

VLLM_PID=$!

echo "Started vLLM PID $VLLM_PID"

sleep 40

while ! curl -s localhost:8000/health >/dev/null 2>&1
do
    # catch if vllm has crashed
    if [ -z "$(ps --pid $VLLM_PID --no-headers)" ]; then
        exit
    fi
    sleep 10
done

echo "vLLM ready"

python classify_mentions_vllm.py

kill $VLLM_PID