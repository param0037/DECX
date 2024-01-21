/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "config.h"


decx::cpuInfo::cpuInfo()
{
    this->is_init = false;
    this->cpu_concurrency = std::thread::hardware_concurrency();
}



_DECX_API_ void de::InitCPUInfo()
{
    decx::cpI.is_init = true;
    decx::cpI.cpu_concurrency = std::thread::hardware_concurrency();
    decx::cpI._hardware_concurrency = std::thread::hardware_concurrency();
}


_DECX_API_ de::DH de::cpu::DecxSetThreadingNum(const size_t _thread_num)
{
    decx::cpI.cpu_concurrency = decx::utils::clamp_min<size_t>(_thread_num, 1);
    de::DH handle;
    if (_thread_num > decx::cpI._hardware_concurrency) {
        decx::warn::CPU_Hyper_Threading(&handle);
    }
    else {
        decx::err::Success(&handle);
    }
    return handle;
}



_DECX_API_ bool decx::cpu::_is_CPU_init()
{
    return decx::cpI.is_init;
}


_DECX_API_ uint64_t decx::cpu::_get_permitted_concurrency()
{
    return decx::cpI.cpu_concurrency;
}


_DECX_API_ uint64_t decx::cpu::_get_hardware_concurrency()
{
    return decx::cpI._hardware_concurrency;
}


// logging configuration


decx::logging_config::logging_config()
{
    this->_enable_log_print = false;
    this->_ignore_successful_print = true;
    this->_ignore_warnings = true;
}


_DECX_API_ bool decx::Get_enable_log_print()
{
    return decx::LogConf._enable_log_print;
}


_DECX_API_ bool decx::Get_ignore_successful_print()
{
    return decx::LogConf._ignore_successful_print;
}


_DECX_API_ bool decx::Get_ignore_warnings()
{
    return decx::LogConf._ignore_warnings;
}


_DECX_API_ void de::DecxEnableLogPrint()
{
    decx::LogConf._enable_log_print = true;
}



_DECX_API_ void de::DecxDisableLogPrint()
{
    decx::LogConf._enable_log_print = false;
}


_DECX_API_ void de::DecxEnableWarningPrint()
{
    decx::LogConf._ignore_warnings = false;
}


_DECX_API_ void de::DecxDisableWarningPrint()
{
    decx::LogConf._ignore_warnings = true;
}


_DECX_API_ void de::DecxEnableSuccessfulPrint()
{
    decx::LogConf._ignore_successful_print = false;
}


_DECX_API_ void de::DecxDisableSuccessfulPrint()
{
    decx::LogConf._ignore_successful_print = true;
}