/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT_COMMONS_H_
#define _FFT_COMMONS_H_

// include basic dependencies
#include "../../core/basic.h"
#include "../../core/utils/double_buffer.h"
#include "../../core/memory_management/PtrInfo.h"
#include "../../core/allocators.h"
#include "../../core/utils/Fixed_Length_Array.h"
#include "../../classes/type_info.h"


#ifdef _DECX_CPU_PARTS_
#include "../../core/allocators.h"
#include "../../core/thread_management/thread_pool.h"
#include "../../core/utils/fragment_arrangment.h"
#include "../../core/thread_management/thread_arrange.h"
#include "../../core/utils/intrinsics_ops.h"
#include "CPU/CPU_FFT_defs.h"
#endif

//
#ifdef _DECX_CUDA_PARTS_
#include "../CUDA_cpf32.cuh"
#include "../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../core/cudaStream_management/cudaStream_queue.h"
#include "../CUDA_cpf32.cuh"
#include "../CUDA_cpd64.cuh"

#define _FFT2D_END_(_cplx_type) false, true, _cplx_type
#define _FFT1D_END_(_cplx_type) _FFT2D_END_(_cplx_type)
#define _IFFT2D_END_(_type_out) true, false, _type_out
#define _IFFT1D_END_(_type_out) false, false, _type_out


namespace decx
{
namespace dsp {
namespace fft {
    typedef struct FFT_krenel_info_for_2D_kernel
    {
        uint32_t _store_pitch;
        uint32_t _warp_proc_len;
        uint32_t _signal_len;

        __host__ __device__ 
        FFT_krenel_info_for_2D_kernel(const uint32_t store_pitch,
                                      const uint32_t warp_proc_len,
                                      const uint32_t signal_len) :
            _store_pitch(store_pitch),
            _warp_proc_len(warp_proc_len),
            _signal_len(signal_len) {}
    }FKI_4_2DK;
}
}
}

#endif

namespace decx
{
namespace dsp {
    namespace fft 
    {
        /**
        * @brief This function is used both in CPU and CUDA.
        *        validate if the type is supported by FFT2D.
        * @param _type The type needs to be validated
        * @return True for valid, false for invalid.
        */
        static bool validate_type_FFT2D(const de::_DATA_TYPES_FLAGS_ _type);


        /**
        * @brief This function is used both in CPU and CUDA.
        *        Check if the two types are matched:
        *        _FP32_ and _COMPLEX_F32_ are matched;
        *        _FP64_ and _COMPLEX_F64_ are matched.
        * @param _type1 The first type
        * @param _type2 The second type
        * @return True for matched, and false for unmatched.
        */
        static bool check_type_matched_FFT(const de::_DATA_TYPES_FLAGS_ _type1,
            const de::_DATA_TYPES_FLAGS_ _type2);
    }
}
}


static bool decx::dsp::fft::validate_type_FFT2D(const de::_DATA_TYPES_FLAGS_ _type)
{
    return (_type < 7 && (_type != 0 || _type != 3 || _type != 4)) ||
        _type == de::_DATA_TYPES_FLAGS_::_UINT8_;
}


static bool decx::dsp::fft::check_type_matched_FFT(const de::_DATA_TYPES_FLAGS_ _type1,
                                                   const de::_DATA_TYPES_FLAGS_ _type2)
{
    return (_type1 & 3) == (_type2 & 3);
}


#define FFT2D_BLOCK_SIZE 16

#define _CUDA_FFT1D_BLOCK_SIZE 512
#define _CUDA_FFT_SHMEM_SIZE_ _CUDA_FFT1D_BLOCK_SIZE * 8  // in byte

#define Pi          3.1415926f
#define Half_Pi     1.5707963f
#define Two_Pi      6.2831853f
#define Four_Pi     12.5663706f
#define Six_Pi      18.8495559f
#define Eight_Pi    25.1327412f


namespace decx
{
namespace dsp {
namespace fft
{
    /**
    * @param _smaller_factors : Ignore radix-4 since 4 = 2 x 2
    * @param __x : Single length
    * @param res_arr : The container to store the radix(es)
    * @return true when the signal length can be apart, false instead
    */
    template <bool _smaller_factors>
    bool _radix_apart(uint64_t __x, std::vector<uint32_t>* res_arr);


    typedef struct _FFT1D_kernel_info FKI1D;


    enum FFT_directions {
        _FFT_AlongH = 0,
        _FFT_AlongW = 1,
        _FFT_AlongD = 2
    };
}
}
}


template <bool _smaller_factors>
static bool decx::dsp::fft::_radix_apart(uint64_t __x, std::vector<uint32_t>* res_arr)
{
    //the order is important
    static uint8_t prime[4] = { 5, 4, 3, 2 };
    uint64_t tmp = 0;

    bool __continue = true;
    bool round_not_f = true;
    
    while (__continue)
    {
        round_not_f = true;
        for (int i = 0; i < 4; ++i)
        {
#ifdef _DECX_CPU_PARTS_
            if constexpr (_smaller_factors) {
                if (i == 1) continue;       // Skip Radix-4
            }
#endif
            if ((__x % prime[i]) == 0) {
                res_arr->push_back(prime[i]);
                round_not_f = false;
                __x /= prime[i];
                break;
            }
        }
        if (round_not_f) {
            __continue = false;
        }
    }
    if (__x != 1) {
        res_arr->insert(res_arr->begin(), __x);
        return false;
    }
    else {
        return true;
    }
}



struct decx::dsp::fft::_FFT1D_kernel_info
{
    uint32_t _radix;
    uint64_t _warp_proc_len;
    uint64_t _signal_len;
    uint64_t _store_pitch;


    _FFT1D_kernel_info(const uint32_t radix, 
                       const uint64_t warp_proc_len, 
                       const uint64_t signal_len, 
                       const uint64_t store_pitch) :
        _radix(radix),
        _warp_proc_len(warp_proc_len),
        _signal_len(signal_len),
        _store_pitch(store_pitch)
    {}


    uint64_t get_warp_num() const
    {
        return this->_signal_len / this->_warp_proc_len;
    }
};


#ifdef _DECX_CUDA_PARTS_

namespace decx
{
    namespace dsp
    {
        static bool check_apart(int __x);
    }
}


static bool decx::dsp::check_apart(int __x)
{
    int prime[4] = { 5, 4, 3, 2 };
    int tmp = 0;
    // ���ж���һ��ȫ���Ҳ������ʵģ�break��whileѭ�����������
    bool __continue = true;
    bool round_not_f = true;

    while (__continue)
    {
        round_not_f = true;
        for (int i = 0; i < 4; ++i) {
            if ((__x % prime[i]) == 0) {
                round_not_f = false;
                __x /= prime[i];
                break;
            }
        }
        if (round_not_f) {    // ���һ����û���ҵ����ʵ�
            __continue = false;
        }
    }
    if (__x != 1) {      // ˵��__x�޷���ȫ�ֽ�
        return false;
    }
    else {
        return true;
    }
}
#endif


#endif