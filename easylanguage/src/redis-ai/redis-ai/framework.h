#pragma once

#define WIN32_LEAN_AND_MEAN             // Windows ヘッダーからほとんど使用されていない部分を除外する
// Windows ヘッダー ファイル
#include <windows.h>
#include <stdlib.h>
#include <time.h>
#include "hiredis/hiredis.h"

#pragma comment( lib, "ws2_32.lib" )