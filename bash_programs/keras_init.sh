#!/bin/bash -l
#PBS -q default
#PBS -A NCCC
#PBS -N Train_Deep_CNN
#PBS -l nodes=4:ppn=8
#PBS -l walltime=168:00:00
#PBS -m abe
#PBS -M ydavidchen.gr@dartmouth.edu

if [ "$HOSTNAME"="x01.hpcc.dartmouth.edu" ]; then
  echo "Using one of the High-Performance Computing (HPC) nodes!"
  cd /global/scratch/ydchen;
  echo $PWD;

  echo "Training CNN network....";
  module load python/3.6-Miniconda;
  source activate DeepLearning3;

  echo "Conda virtual environment activated!";
  python3 trainDeepCNN.py;
  echo "Process complete!";

  source deactivate DeepLearning3;
else
  echo "HPC not detected! Exiting now...";
  exit 1;
fi

exit 0;
