#include "pch.h"
#include "logger.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

int log_write(const char* logfile, const char* format, ...) {
	va_list ap;
	FILE* fp = fopen(logfile, "a");
	if (fp == NULL) {
		return 1;
	}
	else {
		va_start(ap, format);
		vfprintf(fp, format, ap);
		va_end(ap);
	}
	fclose(fp);
	return 0;
}