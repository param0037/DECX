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

decx::cpuInfo decx::cpI;

de::DH decx::_last_error;
#endif