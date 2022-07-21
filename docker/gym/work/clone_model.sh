#!/bin/bash
# 成績の良かったモデルで複数の銘柄モデルを上書きします
SYM_LIST=$1
BEST_HDF5_FILE=$2
MODEL_DIR=$3
BEST_FILE=$(dirname $BEST_HDF5_FILE)/$(basename -s .hdf5 $BEST_HDF5_FILE)
for i in $(cat $SYM_LIST)
do
	cp ${BEST_FILE}.hdf5 ${MODEL_DIR}/${i}.hdf5
	cp ${BEST_FILE}.pkl ${MODEL_DIR}/${i}.pkl
	cp ${BEST_FILE}.pb ${MODEL_DIR}/${i}.pb
	echo "${BEST_FILE} -> ${MODEL_DIR}/${i}"
done
