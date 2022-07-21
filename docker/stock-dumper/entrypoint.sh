#!/bin/bash
/wait-for-it.sh ${RABBITMQ_HOST:-rabbitmq}:${RABBITMQ_PORT:-5672} -- dumper.py
