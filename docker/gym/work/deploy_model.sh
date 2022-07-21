#!/bin/bash
SYM_LIST=$1
MODEL_GROUP=$2
REDIS_HOST=$3
REDIS_HOST=${REDIS_HOST:-redisai}
for i in $(cat $SYM_LIST)
do
    echo deploy.py -n ${i} -g ${MODEL_GROUP} -r ${REDIS_HOST}
done
