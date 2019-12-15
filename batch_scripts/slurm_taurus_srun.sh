#!/bin/bash -l

#SBATCH -p ml
#SBATCH -t 24:00:00
#SBATCH --hint=multithread
#SBATCH --nodes=3
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=28
#SBATCH --mem=0
#SBATCH --gres=gpu:6
#SBATCH --job-name=HVD_2DS
#SBATCH -o experiment1_debug.out

module load modenv/ml
module load OpenMPI/3.1.4-gcccuda-2018b
module load PythonAnaconda/3.6
module load cuDNN/7.1.4.18-fosscuda-2018b
module load CMake/3.11.4-GCCcore-7.3.0

source activate aipp 

echo "JOBID: $SLURM_JOB_ID"
echo "NNODES: $SLURM_NNODES"
echo "NTASKS: $SLURM_NTASKS"
echo "MPIRANK: $SLURM_PROVID"

cd /scratch/p_da_aipp/2D_Schrodinger/2D_Schrodinger

srun --output="run_w_$SLURM_JOB_ID.log" which python3.6

srun --output="run_$SLURM_JOB_ID.log" python3.6 Schrodinger2D_nh_hvd_v2.py --identifier e1_singlenet \
                                --batchsize 10000 \
                                --numbatches 240 \
                                --initsize 7000 \
                                --epochssolution 1000 \
                                --epochsPDE 7000 \
                                --energyloss 0 \
                                --pretraining 1 \
                                --noFeatures 700 \
                                --noLayers 8
