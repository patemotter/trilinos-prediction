#!/bin/bash

export OMP_NUM_THREADS=1

INPUT=$1
OUTDIR=$2
CORES=4

MATDIR=/mnt/usb/just_matrices
EXEDIR=/home/pate/Repos/trilinos-prediction/tpetra_solvers
OPENMPI=/home/pate/openmpi-install

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
        ${OPENMPI}/bin/mpirun -np $CORES ${EXEDIR}/tpetra_solvers ${MATDIR}/${matrix} -d ${OUTDIR} && 
            sed -i "s/${matrix}, -1/${matrix}, 1/g" "$1"
    else
        echo "$COUNT : skipping $matrix"
    fi
done 3< "${INPUT}"
IFS="${OLDIFS}"

