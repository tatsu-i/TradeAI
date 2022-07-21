import os
import time
from logging import getLogger, basicConfig, getLevelName, WARNING
from mackerel.client import Client
from metrics.rabbitmq import rest_queue_list
from metrics.esxi import host_info

logger = getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO")
basicConfig(level=getLevelName(log_level.upper()))

def mackerel_metric(metrics, service_name):
    try:
        client = Client(mackerel_api_key=os.environ.get("MACKEREL_API_KEY"))
        client.post_service_metrics(service_name, metrics)
    except Exception as e:
        logger.error(e)

def main():
    metrics = []
    metrics.append({
        'name': f'custom.metrics.queue_num',
        'time': time.time(), 
        'value': len(rest_queue_list())
    })
    mackerel_metric(metrics, "Stock-AI")
    if os.environ.get("ENABLE_ESXI") == "yes":
        esxi_metrics = host_info(
            host=os.environ.get("ESXI_HOST"),
            port=int(os.environ.get("ESXI_PORT", 443)),
            user=os.environ.get("ESXI_USER", "root"),
            password=os.environ.get("ESXI_PASSWORD", "password"),
        )
        mackerel_metric(esxi_metrics, "ESXi")

if __name__ == "__main__":
    while True:
        main()
        time.sleep(30)
