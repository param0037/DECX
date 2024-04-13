/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "configs/config.h"
#include "thread_management/thread_pool.h"


#ifdef _DECX_CORE_CPU_

_DECX_API_ decx::ThreadPool* decx::thread_pool;

decx::logging_config decx::LogConf;

decx::cpuInfo decx::cpI;
#endif

#ifdef _DECX_ASYNC_CORE_
// Only when user start to add DecxStream, the async_stream_pool starts to run
#include "../../Async Engine/Async_task_threadpool/Async_Engine.h"

decx::async::Async_Engine decx::async::async_engine;
#endif