#include "pch.h"
#include "framework.h"
#include "rabbit-stock.h"
#include "nlohmann/json.hpp"

struct rabbit_connection rc;
using nlohmann::json;

int connect(const char* host, int port, const char* user, const char* password) {

	rc.conn = amqp_new_connection();

	amqp_socket_t* socket = amqp_tcp_socket_new(rc.conn);

	int  rc_sock = amqp_socket_open(socket, host, port);
	if (rc_sock != AMQP_STATUS_OK) {
		return 1;
	}

	amqp_rpc_reply_t  rc_login = amqp_login(
		rc.conn, "/",
		2048,
		AMQP_DEFAULT_FRAME_SIZE,
		AMQP_DEFAULT_HEARTBEAT,
		AMQP_SASL_METHOD_PLAIN,
		user, password);


	if (rc_login.reply_type != AMQP_RESPONSE_NORMAL) {
		return 2;
	}
	rc.channel = 1;
	amqp_channel_open(rc.conn, rc.channel);
	return 0;
}

int disconnect() {
	try {
		amqp_channel_close(
			rc.conn, rc.channel, AMQP_REPLY_SUCCESS
		);
		amqp_connection_close(rc.conn, AMQP_REPLY_SUCCESS);
		amqp_destroy_connection(rc.conn);
	}catch (...) {
		return -1;
	}
	return 0;
}

// symbolname
// feature value: v1 - v11
extern "C" int __declspec(dllexport) __stdcall publish_message(
	const char* host, int port, 
	const char* user, const char* password, 
	const char* symbolname,
	const char* timestamp,
	int currentbar,
	double open,
	double close,
	double high,
	double low,
	double label,
	double v1,
	double v2,
	double v3,
	double v4,
	double v5,
	double v6,
	double v7,
	double v8,
	double v9,
	double v10,
	double v11,
	double v12,
	double v13,
	double v14,
	double v15,
	double v16,
	double v17,
	double v18,
	double v19,
	double v20,
	double v21,
	double v22,
	double v23,
	double v24,
	double v25,
	double v26,
	double v27,
	double v28,
	double v29,
	double v30,
	double v31,
	double v32,
	double v33,
	double v34,
	double v35,
	double v36,
	double v37,
	double v38,
	double v39,
	double v40,
	double v41,
	double v42,
	double v43,
	double v44,
	double v45,
	double v46,
	double v47,
	double v48,
	double v49,
	double v50,
	double v51,
	double v52,
	double v53,
	double v54,
	double v55,
	double v56,
	double v57,
	double v58,
	double v59,
	double v60,
	double v61,
	double v62,
	double v63,
	double v64,
	double v65,
	double v66,
	double v67,
	double v68,
	double v69
)
{
	double feature[] = {
		v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
		v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
		v21, v22, v23, v24, v25, v26, v27, v28, v29, v30,
		v31, v32, v33, v34, v35, v36, v37, v38, v39, v40,
		v41, v42, v43, v44, v45, v46, v47, v48, v49, v50,
		v51, v52, v53, v54, v55, v56, v57, v58, v59, v60,
		v61, v62, v63, v64, v65, v66, v67, v68, v69
	};
	nlohmann::json message;

	amqp_bytes_t  queue;
	int  rc_pub = 0;
	char index[32] = "";

	try {
		if (rc.conn == NULL) {
			connect(host, port, user, password);
		}
		message["symbolname"] = symbolname;
		message["timestamp"] = timestamp;
		message["currentbar"] = currentbar;
		message["open"] = open;
		message["close"] = close;
		message["high"] = high;
		message["low"] = low;
		for (int i = 0; i < sizeof(feature) / sizeof(double); i++) {
			sprintf(index, "Value%d", i);
			message[index] = feature[i];
		}
		queue = amqp_cstring_bytes(symbolname);

		rc_pub = amqp_basic_publish(
			rc.conn, rc.channel,
			amqp_empty_bytes, queue,
			false, false, NULL,
			amqp_cstring_bytes(message.dump().c_str())
		);
	}
	catch (...) {
	}
	return 0;
}

extern "C" int __declspec(dllexport) __stdcall backtest_message(
	const char* host, int port,
	const char* user, 
	const char* password,
	const char* symbolname,
	const char* timestamp,
	const char* expname,
	int netprofit,
	int marketposition,
	int OpenPositionProfit,
	int BarsSinceEntry,
	int action
)
{
	nlohmann::json message;
	amqp_bytes_t  queue;
	int  rc_pub = 0;

	try {
		if (rc.conn == NULL) {
			connect(host, port, user, password);
		}
		message["model_name"] = symbolname;
		message["timestamp"] = timestamp;
		message["expname"] = expname;
		message["netprofit"] = netprofit;
		message["marketposition"] = marketposition;
		message["openpositionprofit"] = OpenPositionProfit;
		message["barssinceentry"] = BarsSinceEntry;
		message["action"] = action;
		queue = amqp_cstring_bytes("backtest");

		rc_pub = amqp_basic_publish(
			rc.conn, rc.channel,
			amqp_empty_bytes, queue,
			false, false, NULL,
			amqp_cstring_bytes(message.dump().c_str())
		);
	}
	catch (...) {
	}
	return 0;
}