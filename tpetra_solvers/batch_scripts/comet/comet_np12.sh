#!/bin/bash

#SBATCH --job-name=UF.1k.np12
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 12

export OMP_NUM_THREADS=1

OUTDIR=$2
MATDIR=~/scratch/UF_Collection_Matrix-Market
EXEDIR=~/projects/trilinos-prediction/tpetra_solvers
INPUT=$1

mkdir -p $OUTDIR

OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read matrix solved <&3
do
    COUNT=$((COUNT + 1))
    if [ "$solved" -ne 1 ]
    then
        echo "$COUNT solving $matrix"
        sed -i "s/${matrix}, 0/${matrix}, -1/g" "$1"
        mpirun ${EXEDIR}/tpetra_solvers ${MATDIR}/${matrix} -d ${OUTDIR} && 
            sed -i "s/${matrix}, -1/${matrix}, 1/g" "$1"
    else
        echo "$COUNT : skipping $matrix"
    fi
done 3< "${INPUT}"
IFS="${OLDIFS}"
