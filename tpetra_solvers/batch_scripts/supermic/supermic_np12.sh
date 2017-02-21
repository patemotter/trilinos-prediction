#!/bin/bash

##SBATCH --job-name=UF.1k.np12
##SBATCH --partition=workq
##SBATCH --time=0:05:00
##SBATCH --nodes 1
##SBATCH --ntasks-per-node 12

#PBS -q workq
#PBS -l nodes=1:ppn=20
#PBS -l walltime=00:05:00


export OMP_NUM_THREADS=1

INPUT=$A
OUTDIR=$B
MATDIR=/home/pmotter/work/UF_Collection_Matrix-Market
EXEDIR=/home/pmotter/work/trilinos-prediction/tpetra_solvers
echo $INPUT
echo $OUTDIR
echo $MATDIR
echo $EXEDIR

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
        mpirun -np 12 ${EXEDIR}/tpetra_solvers ${MATDIR}/${matrix} -d ${OUTDIR} && 
            sed -i "s/${matrix}, -1/${matrix}, 1/g" "$1"
    else
        echo "$COUNT : skipping $matrix"
    fi
done 3< "${INPUT}"
IFS="${OLDIFS}"
