/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/

#include <log_console.h>
#include <stdarg.h>
#include <time.h>
#include <thread_management/global_mutex_array.h>

#ifdef Windows
#include <Windows.h>
#endif
#ifdef Linux
#include <sys/time.h>
#endif


#ifdef Windows
#define SetConsoleColor(_color_flag)    \
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), _color_flag)   \

#define ResetConsoleColor SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 7)
#endif


#ifdef Linux 
#define SetConsoleColor(_color_flag)        \
    printf("\033[0;32;31m");                \

#define ResetConsoleColor printf("\033[m")
#endif


static void DecxSetConsoleTextColor(const DecxInternalLogLevel log_level)
{
#ifdef _MSC_VER
    switch (log_level)
    {
    case LOG_INFO:
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
        break;

    case LOG_NOTICE:
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_BLUE);
        break;

    case LOG_WARNING:
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_GREEN);
        break;

    case LOG_ERROR:
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED);
        break;
    
    default:
        break;
    }
#endif
#if defined(__GNUC__) || defined(__clang__)
    switch (log_level)
    {
    case LOG_INFO:
        break;

    case LOG_NOTICE:
        printf("\033[0;32;34m");
        break;

    case LOG_WARNING:
        printf("\033[0;32;33m");
        break;

    case LOG_ERROR:
        printf("\033[0;32;31m");
        break;
    
    default:
        break;
    }
#endif
}

static void GetSysTimeMsAccurate(char* time_msg, const unsigned int msg_max_length)
{
#ifdef Windows
    SYSTEMTIME sys_time;
    GetSystemTime(&sys_time);
    FILETIME file_time;
    GetSystemTimeAsFileTime(&file_time);
    ULARGE_INTEGER uli_time;

    uli_time.LowPart = file_time.dwLowDateTime;
    uli_time.HighPart = file_time.dwHighDateTime;
    ULONGLONG epoch = uli_time.QuadPart;
    epoch -= 116444736000000000ULL;
    ULONGLONG msec = epoch / 10000;
    snprintf(time_msg, msg_max_length, "%02d:%02d:%02d.%03lld", sys_time.wHour, sys_time.wMinute, sys_time.wSecond, msec % 1000);
#endif
#ifdef Linux
    struct timespec ts;
    struct tm *tm_info;

    clock_gettime(CLOCK_REALTIME, &ts);
    tm_info = localtime(&ts.tv_sec);
    snprintf(time_msg, msg_max_length, "%02d:%02d:%02d.%03ld", tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec, ts.tv_nsec / 1000000);
#endif
}

static void GetSysTimeMsRough(char* time_msg, const unsigned int msg_max_length)
{
#ifdef Windows
    SYSTEMTIME sys_time;
    GetSystemTime(&sys_time);

    snprintf(time_msg, msg_max_length, "%02d:%02d:%02d.%03d", sys_time.wHour, sys_time.wMinute, sys_time.wSecond, sys_time.wMilliseconds);
#endif
#ifdef Linux
    struct timeval tv;
    struct tm* tm_info;

    gettimeofday(&tv, NULL);
    tm_info = localtime(&tv.tv_sec);
    snprintf(time_msg, msg_max_length, "%02d:%02d:%02d.%03ld", tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec, tv.tv_usec / 1000);
#endif
}

void DECX_Log_Console_Exec(const DecxInternalLogLevel log_level, 
                           const char *__restrict module_tag,
                           const char *__restrict func_name, 
                           const char *__restrict msg)
{
    DecxGlobalMtx_Lock(Decx_GMtxId_LogConsole);
    
    char time_info[32];
#if _DECX_CONSOLE_LOG_ENABLE_ACCURATE_TIME_
    GetSysTimeMsAccurate(time_info, 32);
#else
    GetSysTimeMsRough(time_info, 32);
#endif

    DecxSetConsoleTextColor(log_level);

    switch (log_level)
    {
    case LOG_INFO:
        printf("DECX: [%s] <%s::%s> I: %s\n", time_info, module_tag, func_name, msg);
        break;

    case LOG_NOTICE:
        printf("DECX: [%s] <%s::%s> N: %s\n", time_info, module_tag, func_name, msg);
        break;

    case LOG_WARNING:
        printf("DECX: [%s] <%s::%s> W: %s\n", time_info, module_tag, func_name, msg);
        break;

    case LOG_ERROR:
        printf("DECX: [%s] <%s::%s> E: %s\n", time_info, module_tag, func_name, msg);
        break;
    
    default:
        break;
    }

    ResetConsoleColor;
    DecxGlobalMtx_Unlock(Decx_GMtxId_LogConsole);
}
