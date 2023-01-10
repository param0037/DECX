/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "config.h"



_DECX_API_ void de::InitCPUInfo()
{
    decx::cpI.is_init = true;
    decx::cpI.cpu_concurrency = std::thread::hardware_concurrency();
}