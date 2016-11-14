#!/bin/bash

#SBATCH --job-name=UF.1k.np24
#SBATCH --qos=janus-debug
#SBATCH --time=0:20:00
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 3
#SBATCH --output=/lustre/janus_scratch/pamo8800/json_output/slurm-%j.out
#SBATCH --error=/lustre/janus_scratch/pamo8800/json_output/slurm-%j.err

export OMP_NUM_THREADS=4

DATE=`date +%m.%d.%Y_%H:%M`
OUTDIR=~/lustre/
MATDIR=~/lustre/UF_Collection_Matrix-Market
EXEDIR=/home/pamo8800/project/trilinos-prediction/tpetra_solvers
INPUT=/home/pamo8800/project/trilinos-prediction/tpetra_solvers/test_mats.txt

mkdir -p $OUTDIR

OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read matrix solved <&3
do
  if [ $solved -eq 0 ]
  then
    echo solving $matrix
    mpirun $EXEDIR/tpetra_solvers $MATDIR/$matrix -d $OUTDIR
  else
    echo skipping $matrix
  fi
done 3< $INPUT
IFS=$OLDIFS

