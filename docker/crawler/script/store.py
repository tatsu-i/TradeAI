# -*- coding: utf-8 -*-
import os
import sys
import sqlite3
from tqdm import tqdm
from datetime import datetime
from logging import getLogger, basicConfig, getLevelName, WARNING
from elasticsearch import Elasticsearch
from elasticsearch import helpers

logger = getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "WARNING")
basicConfig(level=getLevelName(log_level.upper()))

es_host = os.environ.get("ES_HOST", "elasticsearch")
es_port = int(os.environ.get("ES_PORT", 9200))
es = Elasticsearch(host=es_host, port=es_port)

target_path = sys.argv[1]
connection = sqlite3.connect(target_path)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()
cursor.execute( "select * from emotion" )
actions = []
for row in tqdm(cursor):
    body = dict(row)
    body["timestamp"] = datetime.strptime(body['timestamp'], '%Y-%m-%d %H:%M:%S')
    body["symbolname"] = body["key"].split('_')[1]
    actions.append({'_index':'emotion', '_source':body, '_id': body['key']})
    if len(actions) >= 1000:
        try:
            helpers.bulk(es, actions)
        except Exception as e:
            logger.error(e)
            actions = []
        actions = []
if len(actions) > 0:
    helpers.bulk(es, actions)
connection.close()
