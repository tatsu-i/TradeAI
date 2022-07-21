#!/bin/bash
SYM_LIST=$1
MODEL_GROUP=$2
STEPS=$3
ACTORS=$4
MODEL_GROUP=${MODEL_GROUP:-apex}
STEPS=${STEPS:-30000}
ACTORS=${ACTORS:-3}
for i in $(cat ${SYM_LIST})
do
        echo train.py -f "/data/csv/1.1/${i}.csv" -n ${i} -g ${MODEL_GROUP} -s ${STEPS} -l dqn
done

