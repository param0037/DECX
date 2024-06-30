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


#ifndef _DECX_CPUID_H_
#define _DECX_CPUID_H_


typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;


#ifdef __cplusplus
extern "C" {
#endif
typedef struct _decx_CPUINFO_t
{
    char _vendor_str[12];
    volatile uint32_t _hardware_concurrency;
    volatile uint32_t _base_freq_MHz;
    volatile uint32_t _max_freq_MHz;
    volatile uint64_t _L1_data_cache_size;
    volatile uint64_t _L2_data_cache_size;
    volatile uint64_t _L3_data_cache_size;
}decx_CPUINFO;

#if defined(__x86_64__) || defined(__i386__)
typedef struct _decx_reg4_x86_t
{
    volatile uint32_t _eax;
    volatile uint32_t _ebx;
    volatile uint32_t _ecx;
    volatile uint32_t _edx;
}decx_reg4_x86;

#elif defined(__aarch64__) || defined(__arm__)

#endif

#ifdef _DECX_CORE_CPU_
#ifdef __cplusplus
extern "C"
#endif
void __stdcall CPUID_call(decx_reg4_x86*);
#endif  // #ifdef _DECX_CPU_PARTS_

#if defined(__x86_64__) || defined(__i386__)
uint64_t _decx_get_L1_cache_size_per_phy_core(const int is_AMD)
#ifdef __GNUC__
__attribute__((optimize("O0")))
#else
;
#endif
uint64_t _decx_get_L2_cache_size_per_phy_core(const int is_AMD);
#ifdef __GNUC__
__attribute__((optimize("O0")))
#else
;
#endif
uint64_t _decx_get_L3_cache_size(const int is_AMD);
#ifdef __GNUC__
__attribute__((optimize("O0")))
#else
;
#endif
void _decx_get_CPU_freqs(decx_CPUINFO*);
#ifdef __GNUC__
__attribute__((optimize("O0")))
#else
;
#endif
uint32_t _decx_get_logical_processor_num();
#ifdef __GNUC__
__attribute__((optimize("O0")))
#else
;
#endif
void _decx_get_CPU_vendor_str(decx_CPUINFO*);
#ifdef __GNUC__
__attribute__((optimize("O0")))
#else
;

#endif
#endif      // #if defined(__x86_64__) || defined(__i386__)

int _decx_NOT_AMD_CPU(const decx_CPUINFO*);

int _decx_get_CPU_info(decx_CPUINFO*);

#ifdef __cplusplus
}
#endif

#endif
