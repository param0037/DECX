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

#ifndef _LOG_CONSOLE_H_
#define _LOG_CONSOLE_H_

#include <decx_utils_macros.h>

#define _CONSOLE_LOG_MSG_BUFFER_SIZE_ 256
#define _DECX_CONSOLE_LOG_ENABLE_ACCURATE_TIME_ 0

typedef enum{
    LOG_INFO    = 0,        // White
    LOG_NOTICE  = 1,        // Blue
    LOG_WARNING = 2,        // Yellow
    LOG_ERROR   = 4,        // Red
} DecxInternalLogLevel;


#ifdef __cplusplus
extern "C"{
#endif
    void _DECX_API_ DECX_Log_Console_Exec(const DecxInternalLogLevel color, 
                                          const char* __restrict module_tag,
                                          const char* __restrict func_name, 
                                          const char *__restrict __fmt, ...);
#ifdef __cplusplus
}
#endif

#define DECX_LOG_ERR(...)       DECX_Log_Console_Exec(LOG_ERROR, MODULE_TAG, __FUNCTION__, __VA_ARGS__)
#define DECX_LOG_WARN(...)      DECX_Log_Console_Exec(LOG_WARNING, MODULE_TAG, __FUNCTION__, __VA_ARGS__)
#define DECX_LOG_NOTICE(...)    DECX_Log_Console_Exec(LOG_NOTICE, MODULE_TAG, __FUNCTION__, __VA_ARGS__)
#define DECX_LOG_INFO(...)      DECX_Log_Console_Exec(LOG_INFO, MODULE_TAG, __FUNCTION__, __VA_ARGS__)

#endif