#!/usr/bin/env python3
import os
import sys
import rapidjson
import asyncio
import aio_pika
import time
from datetime import datetime
from pytz import timezone
from logging import getLogger, basicConfig, getLevelName, WARNING
from modules.rabbitmq import RabbitWorker, consumer
from modules.symbol import TOPIX1000
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from mackerel.client import Client

logger = getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO")
basicConfig(level=getLevelName(log_level.upper()))
es_logger = getLogger("elasticsearch")
es_logger.setLevel(WARNING)
pika_logger = getLogger("aio_pika.pika.adapters.base_connection")
pika_logger.setLevel(WARNING)

base_time = 0

def mackerel_metric(name, value):
    global base_time
    if (time.time() - base_time) >= 60:
        base_time = time.time()
    else:
        return
    try:
        client = Client(mackerel_api_key=os.environ.get("MACKEREL_API_KEY"))
        metrics = [
            {
                'name': f'stock_dumper.{name}',
                'time': base_time,
                'value': value
            }
        ]
        client.post_service_metrics('Stock-AI', metrics)
    except Exception as e:
        logger.error(e)

# RabbitMQ Settings
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.environ.get("RABBITMQ_PASS", "guest")
RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", "5672"))
PREFETCH_COUNT = int(os.environ.get("PREFETCH_COUNT", "4096"))
RABBITMQ_IN_QUEUE = os.environ.get("RABBITMQ_IN_QUEUE", "stock")

# elasticsearch
es_host = os.environ.get("ES_HOST", "elasticsearch")
es_port = int(os.environ.get("ES_PORT", 9200))
es = None

BASE_DIR = os.environ.get("BASE_DIR", "/csv")
VERSION = os.environ.get("VERSION", "1.0")
actions = []

def dateparse(timestamp):
    local_tz = timezone('Asia/Tokyo')
    d = datetime.strptime(timestamp, "%Y/%m/%d %H:%M:%S")
    jst = datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    return local_tz.localize(jst)

@consumer
def csv_dumper(message, userdata):
    msg = rapidjson.loads(message)
    mode = "a"
    symbolname = msg['symbolname']
    dumpdir = os.path.join(BASE_DIR, VERSION)
    try:
        os.makedirs(dumpdir)
        logger.info(f"create dir {dumpdir}")
    except:pass
    filename = os.path.join(dumpdir, f"{symbolname}.csv")
    if msg["currentbar"] == 1:
        mode = "w"
        logger.info(f"dump start {filename}")
    try:
        with open(filename, mode) as f:
            if msg["currentbar"] == 1:
                line = ",".join(msg.keys())
                f.write("%s\n" % line)
            line = ",".join(map(str,msg.values()))
            f.write(f"{line}\n")
            mackerel_metric("csv_dumper", 1)
    except:
        mackerel_metric("csv_dumper", 0)

actions = []
@consumer
def es_dumper(message, userdata):
    global actions, es
    try:
        if es is None:
            es = Elasticsearch(host=es_host, port=es_port, timeout=60)
            update_template(es)
    except Exception as e:
        logger.error(e)
        mackerel_metric("es_dumper", 0)
        return

    msg = rapidjson.loads(message)
    msg["@timestamp"] = dateparse(msg["timestamp"])
    del msg["timestamp"]
    expname = msg['expname']
    index = f"backtest-{expname}"
    actions.append({'_index':index, '_source':msg})
    if len(actions) >= 1000:
        try:
            helpers.bulk(es, actions)
        except Exception as e:
            logger.error(e)
            mackerel_metric("es_dumper", 0)
            actions = []
            es = None
            return
        actions = []
    mackerel_metric("es_dumper", 1)

def update_template(es):
    template = {
      "order": 0,
      "index_patterns": [
        "backtest*"
      ],
      "settings": {
          "index" :{
              "number_of_replicas": 0,
              "refresh_interval": "15s"
          }
      },
      "aliases": {}
    }
    es.indices.put_template("backtest", template)
    logger.info("update elasticsearch template")

if __name__ == "__main__":
    rw = RabbitWorker(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        user=RABBITMQ_USER,
        password=RABBITMQ_PASS,
    )
    rw.add_consumer(
        consumer=es_dumper(
            in_queue="backtest",
            prefetch_count=3000,
            userdata=""
        ),
        worker_num=1,
    )
    for symbolname in TOPIX1000:
        logger.info(symbolname)
        rw.add_consumer(
            consumer=csv_dumper(
                in_queue=symbolname,
                prefetch_count=3000,
                userdata=""
            ),
            worker_num=1,
        )
    rw.run_forever()
