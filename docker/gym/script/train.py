import os
import sys
import json
import pika
import click

RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.environ.get("RABBITMQ_PASS", "guest")
RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", "5672"))

def publish(message, queue_name, algo):
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)
    channel.basic_publish(
        exchange='gym',
        routing_key=algo,
        body=json.dumps(message),
    )
    connection.close()

@click.command()
@click.option('--csvfile', '-f')
@click.option('--model_name', '-n', default='default')
@click.option('--model_group', '-g', default='r2d2')
@click.option('--nb_steps', '-s', default=30000, type=int)
@click.option('--num_actors', '-a', default=6, type=int)
@click.option('--algo', '-l', type=click.Choice(['apex', 'r2d2', 'dqn']), default="apex")
@click.option('--mode', type=click.Choice(['stg', 'prod']), default='stg')
def main(csvfile, model_name, model_group, nb_steps, num_actors, algo, mode):
    msg = {}
    msg["csvfile"] = csvfile
    msg["model_name"] = model_name
    msg["model_group"] = model_group
    msg["nb_steps"] = nb_steps
    msg["num_actors"] = num_actors
    msg["mode"] = mode
    print(msg)
    publish(msg, f"{algo}_trainer", algo)

if __name__ == '__main__':
    main()
