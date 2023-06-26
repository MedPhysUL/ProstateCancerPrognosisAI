#!/bin/bash
#SBATCH --job-name=MaxenceMETASTASIS
#SBATCH --partition=gpu_96h
#SBATCH --cpus-per-task=16
#SBATCH --time=4-00:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxence.larose.1@ulaval.ca

nvidia-smi

module load apptainer

apptainer run -C --nv --bind /home/ulaval.ca/malar507/ProstateCancerPrognosisAI/experiments/:/workspace/applications/experiments,/home/ulaval.ca/malar507/ProstateCancerPrognosisAI/temp/:/workspace/applications/temp prognosis_unextractor.sif main_METASTASIS.py