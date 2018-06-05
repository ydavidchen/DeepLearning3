#!/bin/bash -l
# submissionPacket.sh
# Make submission packet consisting of required files

echo "Specifying path and making new directory...";
cd /global/scratch/ydchen;
mkdir DavidChen_submission;
echo "Making copies of required files...";
cp trainDeepCNN.py DavidChen_submission/; #code folder
cp DL3\ Dataset/outputs/submit_DavidChen.csv DavidChen_submission/; #submission file
cp Train_Deep_CNN.o* DavidChen_submission/; #output log
echo "Ready for download!";
echo "Once donwloaded to local machine, put in additional files and compress into single file";
