#pragma once
#include "pch.h"

struct redis_connection {
	redisContext* conn;
};

extern struct redis_connection rc;

int connect(const char*, int);

int disconnect(void);