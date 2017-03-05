#!/bin/bash

#SBATCH --job-name=UF.1k.np4
#SBATCH -p RM
#SBATCH --time=12:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4

export OMP_NUM_THREADS=1

OUTDIR=$2
MATDIR=~/pylon2/UF_Collection_Matrix-Market
EXEDIR=~/pylon2/trilinos-prediction/tpetra_solvers
INPUT=$1

mkdir -p $OUTDIR
set -x

IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read matrix solved <&3
do
    mpirun ${EXEDIR}/tpetra_solvers ${MATDIR}/${matrix} -d ${OUTDIR} 
done 3< $INPUT
