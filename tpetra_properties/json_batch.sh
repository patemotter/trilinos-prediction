#!/bin/bash

###
#       Executes program (properties/solver) on matrices using a csv 
#       passed in as an argument. The csv consists of a matrix name and
#       binary entry indicating if it has been processed or not. 
#       Ex: "test.mtx, 0" This 0 gets updated to a 1 upon successfully
#       completing the execution. Else it remains at 0. 
###

OUTDIR=/media/sf_F_DRIVE/output
MATDIR=/media/sf_E_DRIVE/UFget/untarred_mtx
EXEDIR=/media/sf_F_DRIVE/repos/trilinos-prediction/tpetra_properties

INPUT=$1
COUNT=0

mkdir -p $OUTDIR

OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read matrix solved <&3
do
    COUNT=$((COUNT + 1))
    if [ "$solved" -eq 0 ]
    then
        echo "$COUNT solving $matrix"
        mpirun -np 4 $EXEDIR/tpetra_properties_crsmatrix $MATDIR/$matrix -json $OUTDIR/out.json && sed -i "s/${matrix}, 0/${matrix}, 1/g" "$1"
    else
        echo "$COUNT : skipping $matrix"
    fi
done 3< $INPUT
IFS=$OLDIFS
