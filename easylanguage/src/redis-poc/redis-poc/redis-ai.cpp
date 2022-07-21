// stock-ai.cpp : DLL 用にエクスポートされる関数を定義します。
//

#include "pch.h"
#include "framework.h"
#include "redis-ai.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>

struct redis_connection rc;
using namespace std;


time_t strptime(const char* timebuf, const char* fmt, struct tm* timeptr)
{
	int res;
	int year, month, day, hour, minute, second = 0;

	res = sscanf(timebuf, fmt,
		&year, &month, &day,
		&hour, &minute, &second);
	if (res < 6) {
		return -1;
	}
	timeptr->tm_sec = second;
	timeptr->tm_min = minute;
	timeptr->tm_hour = hour;
	timeptr->tm_mday = day;
	timeptr->tm_mon = month - 1;
	timeptr->tm_year = year - 1900;
	timeptr->tm_isdst = -1;
	return 0;
}

double Julian(int y, int m, int d, int H, int M)
{
	int mm, yy;
	int k1, k2, k3;
	double j;

	yy = y - (int)((12 - m) / 10);
	mm = m + 9;
	if (mm >= 12)
	{
		mm = mm - 12;
	}
	k1 = (int)(365.25 * (yy + 4712));
	k2 = (int)(30.6001 * mm + 0.5);
	k3 = (int)((int)((yy / 100) + 49) * 0.75) - 38;
	// 'j' for dates in Julian calendar:
	j = k1 + k2 + d + 59 + (H / 24.0) + (M / 3600.0);
	if (j > 2299160)
	{
		// For Gregorian calendar:
		j = j - k3; // 'j' is the Julian date at 12h UT (Universal Time)
	}
	return j;
}

int MoonAge(int d, int m, int y, int H, int M)
{

	double ag = 0.0;
	double ip = 0.0;
	double j = Julian(d, m, y, H, M);
	//Calculate the approximate phase of the moon
	ip = (j + 4.867) / 29.53059;
	ip = ip - floor(ip);
	//After several trials I've seen to add the following lines, 
	//which gave the result was not bad 
	if (ip < 0.5)
		ag = ip * 29.53059 + 29.53059 / 2;
	else
		ag = ip * 29.53059 - 29.53059 / 2;
	// Moon's age in days
	ag = floor(ag);
	return (int)ag;
}

int GetDays(int y, int m, int d)
{
	// 1・2月 → 前年の13・14月
	if (m <= 2)
	{
		--y;
		m += 12;
	}
	int dy = 365 * (y - 1); // 経過年数×365日
	int c = y / 100;
	int dl = (y >> 2) - c + (c >> 2); // うるう年分
	int dm = (m * 979 - 1033) >> 5; // 1月1日から m 月1日までの日数
	return dy + dl + dm + d - 1;
}

int destiny(int y, int m, int d) {
	int result = 0;
	result = (GetDays(y, m, d) + 8) % 12;
	return result;
}

int to_categorical(int* feature, int category, int num_class) {
	for (int i = 0; i < num_class; i++) {
		if (category == i) {
			feature[i] = 1;
		}
		else {
			feature[i] = 0;
		}
	}
	return 0;
}

int connect(const char* host, int port) {
	// Redis 接続
	rc.conn = redisConnect(
		host, // 接続先redisサーバ
		port  // ポート番号
	);
	if ((NULL != rc.conn) && rc.conn->err) {
		return 1;
	}
	else if (NULL == rc.conn) {
		return 2;
	}
	return 0;
}

int disconnect() {
	redisFree(rc.conn);
	rc.conn = NULL;
	return 0;
}

double _predict(
	double* result,
	const char* redis_host,
	int redis_port,
	int currentBar,
	const char* symbolname,
	const char* timestamp,
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
	int input_dim = 135;
	char tid[254] = "";
	double max_output = 0.0;
	double output = 0.0;
	int i = 0;
	int features[66];
	struct tm res;
	*result = 0;

	sprintf(tid, "%lu", GetCurrentThreadId());
	redisReply* resp = NULL;

	try {
		if ((NULL != rc.conn) && rc.conn->err) {
			redisFree(rc.conn);
			rc.conn = NULL;
		}
		if (NULL == rc.conn) {
			if (connect(redis_host, redis_port) != 0) {
				return -1;
			}
		}
		strptime(timestamp, "%d/%d/%d %d:%d:%d", &res);
		to_categorical(&features[0], res.tm_hour, 24);
		to_categorical(&features[24], destiny(res.tm_year + 1900, res.tm_mon + 1, res.tm_mday), 12);
		to_categorical(&features[36], MoonAge(res.tm_year + 1900, res.tm_mon + 1, res.tm_mday, res.tm_hour, res.tm_min), 30);
		freeReplyObject((redisReply*)redisCommand(rc.conn,
			"AI.TENSORSET input_%s FLOAT 1 %d VALUES %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
			tid, input_dim,
			v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
			v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
			v21, v22, v23, v24, v25, v26, v27, v28, v29, v30,
			v31, v32, v33, v34, v35, v36, v37, v38, v39, v40,
			v41, v42, v43, v44, v45, v46, v47, v48, v49, v50,
			v51, v52, v53, v54, v55, v56, v57, v58, v59, v60,
			v61, v62, v63, v64, v65, v66, v67, v68, v69,
			features[0], features[1], features[2], features[3], features[4],
			features[5], features[6], features[7], features[8], features[9],
			features[10], features[11], features[12], features[13], features[14],
			features[15], features[16], features[17], features[18], features[19],
			features[20], features[21], features[22], features[23], features[24],
			features[25], features[26], features[27], features[28], features[29],
			features[30], features[31], features[32], features[33], features[34],
			features[35], features[36], features[37], features[38], features[39],
			features[40], features[41], features[42], features[43], features[44],
			features[45], features[46], features[47], features[48], features[49],
			features[50], features[51], features[52], features[53], features[54],
			features[55], features[56], features[57], features[58], features[59],
			features[60], features[61], features[62], features[63], features[64],
			features[65]
		));
		freeReplyObject((redisReply*)redisCommand(rc.conn,
			"AI.MODELRUN %s INPUTS input_%s OUTPUTS output_%s", symbolname, tid, tid));
		resp = (redisReply*)redisCommand(rc.conn, "AI.TENSORGET output_%s VALUES", tid);
		if (NULL == resp)
			return -2;
		if (REDIS_REPLY_ERROR == resp->type) {
			freeReplyObject(resp);
			return -3;
		}
		if (resp->type == REDIS_REPLY_ARRAY) {
			if (resp->element[2]->type == REDIS_REPLY_ARRAY) {
				for (i = 0; i < resp->element[2]->elements; i++) {
					output = atof(resp->element[2]->element[i]->str);
					if (max_output < output) {
						max_output = output;
						*result = (double)i;
					}
				}
			}
		}
		freeReplyObject(resp);
		freeReplyObject((redisReply*)redisCommand(rc.conn, "DEL input_%s output_%s", tid, tid));
		if (currentBar == 1) {
			disconnect();
		}
	}
	catch (...) {
	}
	return 0;
}

extern "C" double __declspec(dllexport) __stdcall predict(
	const char* redis_host,
	int redis_port,
	int currentBar,
	const char* symbolname,
	const char* timestamp,
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
) {
	double result = 0;
	int r = 0;
	r = _predict(&result,
		redis_host, redis_port, currentBar, symbolname, timestamp,
		v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
		v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
		v21, v22, v23, v24, v25, v26, v27, v28, v29, v30,
		v31, v32, v33, v34, v35, v36, v37, v38, v39, v40,
		v41, v42, v43, v44, v45, v46, v47, v48, v49, v50,
		v51, v52, v53, v54, v55, v56, v57, v58, v59, v60,
		v61, v62, v63, v64, v65, v66, v67, v68, v69
	);
	if (r != 0) {
		ofstream outputfile("C:\\stock-ai-error.log", ios_base::out | ios_base::app);
		outputfile << "symbolname:" << symbolname << " predict error" << r << endl;
		outputfile.close();
	}
	return result;
}