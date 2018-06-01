#!/bin/bash -l
#PBS -q default
#PBS -A NCCC
#PBS -N Train_Deep_CNN
#PBS -l nodes=4:ppn=8
#PBS -l walltime=168:00:00
#PBS -m abe
#PBS -M ydavidchen.gr@dartmouth.edu

echo "Training CNN network....";
cd /global/scratch/ydchen;
echo $PWD;
module load python/3.6-Miniconda;
source activate DeepLearning3;
echo "Virtual environment activated!";
python3 trainDeepCNN.py;
echo "Process complete!";

source deactivate DeepLearning3;
exit 0;
