#!/bin/bash
set -e

#this script launches jobs.xml from given dir

DIRNAME=$1

MODEL="blahblah"

if test -d $DIRNAME
then
    for i in $DIRNAME*.slurm
    do
        # echo -e "\n Launching $i !"
        eval "sbatch $MODEL $i 8 --time=24:00:00"
    done
else
	echo "directory doesnt exists"
fi

echo -e "\n All jobs from $DIRNAME launched!"
