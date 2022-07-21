#!/bin/bash
# 成績の良かったモデルをbestモデルとしてコピーします。
DIR=${1:-staging}
mkdir -p /data/model/${DIR}
awk -v DIR=$DIR -F "," 'BEGIN{RS="\r\n"}{print "cp /data/model/"$2"/"$1".hdf5 /data/model/"DIR"/"$1".hdf5"}' all.csv
awk -v DIR=$DIR -F "," 'BEGIN{RS="\r\n"}{print "cp /data/model/"$2"/"$1".pkl /data/model/"DIR"/"$1".pkl"}' all.csv
awk -v DIR=$DIR -F "," 'BEGIN{RS="\r\n"}{print "cp /data/model/"$2"/"$1".pb /data/model/"DIR"/"$1".pb"}' all.csv

