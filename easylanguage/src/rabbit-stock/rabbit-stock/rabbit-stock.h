#pragma once
#include <amqp.h>
#include <amqp_tcp_socket.h>
#include <amqp_framing.h>

struct rabbit_connection {
	amqp_connection_state_t  conn;
	amqp_channel_t  channel;
};

extern struct rabbit_connection rc;

int connect(const char* , int, const char* , const char*);

int disconnect(void);