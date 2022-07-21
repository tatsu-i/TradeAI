#!/bin/bash
# バックテスト結果をCSVに出力します
aggs-models.py -e staging -e production -e test -d 14
cut poor.csv -d ',' -f1 > symbol/poor.txt
cut rich.csv -d ',' -f1 > symbol/rich.txt
cut all.csv -d ',' -f1 > symbol/all.txt
