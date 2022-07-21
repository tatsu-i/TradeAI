#!/bin/bash
cd /data
tar zcpvf /mnt/TradeStation/backup-$(date +"%Y%m%d%H%M").tar.gz model csv
