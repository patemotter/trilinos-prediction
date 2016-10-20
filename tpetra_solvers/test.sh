#!/bin/bash
OUTDIR=/media/sf_F_DRIVE/output
MATDIR=/media/sf_E_DRIVE/UFget/untarred_mtx
EXEDIR=/media/sf_F_DRIVE/repos/trilinos-prediction/tpetra_properties

INPUT=$1

mkdir -p $OUTDIR

OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read matrix solved <&3
do
    if [ $solved -eq 0 ]
    then
        echo $EXEDIR
        echo $MATDIR
        echo $matrix
        $EXEDIR/tpetra_properties_crsmatrix $MATDIR/$matrix -json $OUTDIR/out.json && sed -i "s/${matrix}, 0/${matrix}, 1/g" "$1"
    else
        echo skipping $matrix
    fi
done 3< $INPUT
IFS=$OLDIFS
