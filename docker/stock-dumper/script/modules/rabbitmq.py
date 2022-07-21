import os
import types
import logging
import asyncio
import pika
import multiprocessing
from retry import retry
from functools import wraps

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("pika").propagate = False
PREFETCH_COUNT = 1

class RabbitWorker:
    def __init__(self, host, port=5672, user="guest", password="guest"):
        self.url = "amqp://%s:%s@%s:%s/" % (user, password, host, port)
        self.consumers = []

    def add_consumer(self, consumer, worker_num=1):
        self.consumers.append(
            {
                "consumer": consumer,
                "worker_num": worker_num,
            }
        )

    async def consumer(self, consumer):
        await consumer(self.url)

    @retry(tries=9, delay=6, backoff=3)
    def run_consumer(self, consumer):
        loop = asyncio.get_event_loop()
        loop.create_task(
            self.consumer(consumer)
        )
        loop.run_forever()

    def run_forever(self):
        # consumers
        for f in self.consumers:
            for i in range(f["worker_num"]):
                multiprocessing.Process(
                    target=self.run_consumer,
                    args=(
                        f["consumer"],
                    ),
                ).start()

def consumer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        async def connect(url):
            in_queue = kwargs["in_queue"]
            prefetch_count = kwargs.get("prefetch_count", PREFETCH_COUNT)
            durable = kwargs.get("durable", False)
            userdata = kwargs.get("userdata")
            timeout = kwargs.get("timeout", 600)
            # connect rabbitmq
            exchange_name = 'stock-ai'
            parameters = pika.URLParameters(url)
            parameters.heartbeat = 0
            parameters.socket_timeout=10
            parameters.blocked_connection_timeout=120
            connection = pika.BlockingConnection(parameters)
            def on_message(channel, method_frame, header_frame, body):
                try:
                    func(body, userdata)
                    channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                except Exception as e:
                    logger.exception(e)
            channel = connection.channel()
            channel.basic_qos(prefetch_count=prefetch_count)
            channel.exchange_declare(exchange=exchange_name,exchange_type='direct', durable=False)
            channel.queue_declare(queue=in_queue, auto_delete=False, durable=False)
            channel.queue_bind(queue=in_queue, exchange=exchange_name, routing_key='')
            channel.basic_consume(in_queue, on_message, consumer_tag=os.uname()[1])
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
            connection.close()
        return connect

    return wrapper
