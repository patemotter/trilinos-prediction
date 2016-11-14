#!/bin/bash
OUTDIR=/home/pate/Repos/trilinos-prediction/tpetra_solvers
MATDIR=/home/pate/Repos/trilinos-prediction/test_matrices
EXEDIR=/home/pate/Repos/trilinos-prediction/tpetra_solvers

INPUT=$1
PROCS=$2
OMP_NUM_THREADS=$3

mkdir -p $OUTDIR

OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read matrix solved <&3
do
  if [ $solved -eq 0 ]
  then
    echo solving $matrix
    mpirun -n $PROCS $EXEDIR/tpetra_solvers $MATDIR/$matrix -d $OUTDIR && 
    sed -i "s/${matrix}, 0/${matrix}, 1/g" "$1"
  else
    echo skipping $matrix
  fi
done 3< $INPUT
IFS=$OLDIFS
