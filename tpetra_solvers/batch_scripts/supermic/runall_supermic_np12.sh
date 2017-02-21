#!/bin/bash

qsub -v A="/home/pmotter/work/trilinos-prediction/matrix_lists/supermic/np12/UF_1000_split.aa",B="/home/pmotter/work/results/np12_results_aa" ./supermic_np12.sh 
