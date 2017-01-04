#!/bin/bash

#SBATCH --job-name=UF.1k.np10
#SBATCH --qos=janus
###SBATCH --time=12:00:00
#SBATCH --time=4:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 10
#SBATCH --output=/lustre/janus_scratch/pamo8800/output/slurm-%j.out
#SBATCH --error=/lustre/janus_scratch/pamo8800/error/slurm-%j.err

export OMP_NUM_THREADS=1

module load slurm

OUTDIR=$2
MATDIR=~/lustre/UF_Collection_Matrix-Market
EXEDIR=/home/pamo8800/project/trilinos-prediction/tpetra_solvers
INPUT=$1

mkdir -p $OUTDIR
export I_MPI_PMI_LIBRARY=/curc/slurm/slurm/current/lib/libpmi.so

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
