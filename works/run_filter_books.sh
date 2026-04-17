#!/bin/bash
#SBATCH --account=project_2017429
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00

source /scratch/project_2017429/chiunhau/my_python_env/bin/activate

cd /scratch/project_2017429/chiunhau/birds/works

python filter_books_by_keywords.py --workers $SLURM_CPUS_PER_TASK
