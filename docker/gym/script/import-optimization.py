import sys,os
import json
import csv
from elasticsearch import Elasticsearch
from elasticsearch import helpers

csvfile = sys.argv[1]
es_host = sys.argv[2]
es_port = int(sys.argv[3])
es = Elasticsearch(host=es_host, port=es_port)

with open(csvfile, 'r') as f:
    actions = []
    model_name, ext = os.path.splitext(os.path.basename(csvfile))
    for row in csv.DictReader(f):
        doc = {}
        row = dict(row)
        for k, v in row.items():
            try:
                v = float(v)
            except:pass
            doc[k] = v
        actions.append({'_index':f'optimization-{model_name}', '_source':doc})
    helpers.bulk(es, actions)
