/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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

#ifdef __cplusplus
extern "C"
#endif
void _stdcall CPUID_call(decx_reg4_x86*);

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

int _decx_get_CPU_info(decx_CPUINFO*);

#ifdef __cplusplus
}
#endif

#endif
