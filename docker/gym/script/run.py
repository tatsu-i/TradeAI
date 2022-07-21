import os
import sys
import json
import pika
import click
import multiprocessing
from retry import retry
from model.dqn import train_dqn
from model.r2d2 import train_r2d2
from model.apex import train_apex
from keras.models import load_model
from logging import getLogger, basicConfig, getLevelName, WARNING
logger = getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO")
basicConfig(level=getLevelName(log_level.upper()))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# RabbitMQ Settings
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.environ.get("RABBITMQ_PASS", "guest")
RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", "5672"))

def dqn_trainer(channel, method_frame, header_frame, body):
    msg = json.loads(body)
    csvfile = msg["csvfile"]
    model_name = msg["model_name"]
    model_group = msg["model_group"]
    nb_steps = msg["nb_steps"]
    num_actors = 12
    mode = msg.get("mode", "stg")
    logger.info(f"train {model_name} started.")
    train_dqn(csvfile, model_name, model_group, nb_steps, num_actors, mode)
    logger.info(f"train {model_name} finished.")
    channel.basic_ack(delivery_tag=method_frame.delivery_tag)

def apex_trainer(channel, method_frame, header_frame, body):
    msg = json.loads(body)
    csvfile = msg["csvfile"]
    model_name = msg["model_name"]
    model_group = msg["model_group"]
    nb_steps = msg["nb_steps"]
    num_actors = 12
    mode = msg.get("mode", "stg")
    logger.info(f"train {model_name} started.")
    train_apex(csvfile, model_name, model_group, nb_steps, num_actors, mode)
    logger.info(f"train {model_name} finished.")
    channel.basic_ack(delivery_tag=method_frame.delivery_tag)

def r2d2_trainer(channel, method_frame, header_frame, body):
    msg = json.loads(body)
    csvfile = msg["csvfile"]
    model_name = msg["model_name"]
    model_group = msg["model_group"]
    nb_steps = msg["nb_steps"]
    num_actors = 12
    mode = msg.get("mode", "stg")
    logger.info(f"train {model_name} started.")
    train_r2d2(csvfile, model_name, model_group, nb_steps, num_actors, mode)
    logger.info(f"train {model_name} finished.")
    channel.basic_ack(delivery_tag=method_frame.delivery_tag)

@retry(tries=9, delay=6, backoff=3)
def trainer_worker(algo):
    while True:
        try:
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
            parameters = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                heartbeat = 0,
                credentials=credentials,
                socket_timeout=10,
                blocked_connection_timeout=120
            )
            trainer = f"{algo}_trainer"
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.basic_qos(prefetch_count=1)
            channel.exchange_declare(exchange='gym',exchange_type='direct', durable=True)
            channel.queue_declare(queue=trainer, auto_delete=False, durable=True)
            channel.queue_bind(queue=trainer, exchange='gym', routing_key=algo)
            channel.basic_consume(trainer, globals()[trainer], consumer_tag=os.uname()[1])
            channel.start_consuming()
        except Exception as e:
            logger.error(e)

@click.command()
@click.option('--algo', '-l', type=click.Choice(['apex', 'r2d2', 'dqn']), default="dqn")
@click.option('--worker', '-w', type=int, default=6)
def main(algo, worker):
    for i in range(worker):
        multiprocessing.Process(
            target=trainer_worker,
            args=(algo,),
        ).start()

if __name__ == '__main__':
    main()
