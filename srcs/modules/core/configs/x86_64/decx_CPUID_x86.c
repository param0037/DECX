/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../decx_CPUID.h"

#ifdef _MSC_VER
#pragma optimize("", off)
#endif
uint64_t _decx_get_L1_cache_size_per_phy_core(const int not_AMD)
{
    decx_reg4_x86 regs;
    regs._eax = (not_AMD == 0 ? 0x8000001D : 0x04);
    regs._ecx = 0x01;
    CPUID_call(&regs);

    uint64_t L1 = 0;
    L1 = (regs._ebx & 4095U) + 1U;
    L1 *= (((regs._ebx >> 12) & 0x3FFU) + 1U);
    L1 *= (((regs._ebx >> 22) & 0x3FFU) + 1U);
    L1 *= (regs._ecx + 1U);

    return L1;
}


uint64_t _decx_get_L2_cache_size_per_phy_core(const int not_AMD)
{
    decx_reg4_x86 regs;
    regs._eax = (not_AMD == 0 ? 0x8000001D : 0x04);
    regs._ecx = 0x02;
    CPUID_call(&regs);

    uint64_t L2 = 0;
    L2 = (regs._ebx & 4095U) + 1U;
    L2 *= (((regs._ebx >> 12) & 0x3FFU) + 1U);
    L2 *= (((regs._ebx >> 22) & 0x3FFU) + 1U);
    L2 *= (regs._ecx + 1U);

    return L2;
}


uint64_t _decx_get_L3_cache_size(const int not_AMD)
{
    decx_reg4_x86 regs;
    regs._eax = (not_AMD == 0 ? 0x8000001D : 0x04);
    regs._ecx = 0x03;
    CPUID_call(&regs);

    uint64_t L3 = 0;
    L3 = (regs._ebx & 4095U) + 1U;
    L3 *= (((regs._ebx >> 12) & 0x3FFU) + 1U);
    L3 *= (((regs._ebx >> 22) & 0x3FFU) + 1U);
    L3 *= (regs._ecx + 1U);

    return L3;
}


void _decx_get_CPU_freqs(decx_CPUINFO* _info_ptr)
{
    decx_reg4_x86 regs;
    regs._eax = 0x16;
    CPUID_call(&regs);
    _info_ptr->_base_freq_MHz = regs._eax;
    _info_ptr->_max_freq_MHz = regs._ebx;
}


uint32_t _decx_get_logical_processor_num()
{
    decx_reg4_x86 regs;
    regs._eax = 0x0B;
    regs._ecx = 1;
    CPUID_call(&regs);
    return regs._ebx;
}


void _decx_get_CPU_vendor_str(decx_CPUINFO* _info_ptr)
{
    decx_reg4_x86 regs;
    regs._eax = 0x00;
    CPUID_call(&regs);
    ((uint32_t*)_info_ptr->_vendor_str)[0] = regs._ebx;
    ((uint32_t*)_info_ptr->_vendor_str)[1] = regs._edx;
    ((uint32_t*)_info_ptr->_vendor_str)[2] = regs._ecx;
}


int _decx_get_CPU_info(decx_CPUINFO* _info_ptr)
{
    if (_info_ptr) {
        _decx_get_CPU_vendor_str(_info_ptr);
        const int _not_AMD = (*((uint32_t*)_info_ptr->_vendor_str) ^ 1752462657);
        _info_ptr->_hardware_concurrency = _decx_get_logical_processor_num();
        _info_ptr->_L1_data_cache_size = _decx_get_L1_cache_size_per_phy_core(_not_AMD);
        _info_ptr->_L2_data_cache_size = _decx_get_L2_cache_size_per_phy_core(_not_AMD);
        _info_ptr->_L3_data_cache_size = _decx_get_L3_cache_size(_not_AMD);
        _decx_get_CPU_freqs(_info_ptr);

        return 0;
    }
    else {
        return -1;
    }
}
#ifdef _MSC_VER
#pragma optimize("", on)
#endif
