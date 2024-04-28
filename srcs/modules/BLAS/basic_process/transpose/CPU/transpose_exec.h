/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _TRANSPOSE_EXEC_H_
#define _TRANSPOSE_EXEC_H_


#include "../../../../core/basic.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../../../core/configs/config.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../core/utils/intrinsics_ops.h"


namespace decx
{
    namespace bp 
    {
        template <uint8_t _element_byte>
        struct _cpu_transpose_config;


        template <uint8_t _element_byte>
        struct _cpu_transpose_MK_config;


        namespace CPUK 
        {
#ifdef __GNUC__
            _THREAD_CALL_ static inline void
            block8x8_transpose_u8(__m64 _regs0[8], __m64 _regs1[8]);
#endif
#ifdef _MSC_VER
            _THREAD_CALL_ static inline void
            block8x8_transpose_u8(uint64_t _regs0[8], uint64_t _regs1[8]);
#endif


            /**
            * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
            * @param Wsrc : width of source matrix (in vec4)
            * @param Wdst : height of destinated matrix (in vec4)
            */
            _THREAD_FUNCTION_ void
            transpose_4x4_b32(const float* src, float* dst, const uint2 proc_dims_src, const uint32_t Wsrc_v1, const uint32_t Wdst_v1);


            /**
            * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
            * @param Wsrc : width of source matrix (in vec4)
            * @param Wdst : height of destinated matrix (in vec4)
            */
            _THREAD_FUNCTION_ void
            transpose_4x4_b32_LH(const float* src, float* dst, const uint2 proc_dims_src, const uint32_t Wsrc_v1, const uint32_t Wdst_v1);


            /**
            * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
            * @param Wsrc : width of source matrix (in vec4)
            * @param Wdst : height of destinated matrix (in vec4)
            */
            _THREAD_FUNCTION_ void
            transpose_8x8_b8(const double* src, double* dst, const uint2 proc_dims_src, const uint32_t Wsrc_v8, const uint32_t Wdst_v8);


            /**
            * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
            * @param Wsrc : width of source matrix (in vec4)
            * @param Wdst : height of destinated matrix (in vec4)
            */
            _THREAD_FUNCTION_ void
            transpose_8x8_b8_LH(const double* src, double* dst, const uint2 proc_dims_src, const uint32_t Wsrc_v8, const uint32_t Wdst_v8);



            /**
            * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
            * @param Wsrc : width of source matrix (in vec4)
            * @param Wdst : height of destinated matrix (in vec4)
            */
            _THREAD_FUNCTION_ void
            transpose_2x2_b64(const double* src, double* dst, const uint2 proc_dims_src, const uint32_t Wsrc_v1, const uint32_t Wdst_v1);



            /**
            * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
            * @param Wsrc : width of source matrix (in vec4)
            * @param Wdst : height of destinated matrix (in vec4)
            */
            _THREAD_FUNCTION_ void
            transpose_2x2_b64_LH(const double* src, double* dst, const uint2 proc_dims_src, const uint32_t Wsrc, const uint32_t Wdst);


            _THREAD_FUNCTION_ void
            transpose_MK_2x2_b64_LH(const double* src, double* dst, const uint2 proc_dims_src, const uint32_t Wsrc, const uint32_t Wdst, 
                const uint32_t _ch_num, const uint64_t _gapsrc_v1, const uint64_t _gapdst_v1);
        }


        /**
        * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
        * @param Wsrc : width of source matrix (in element)
        * @param Wdst : height of destinated matrix (in element)
        */
        void transpose_8x8_caller(const double* src, double* dst, const uint32_t Wsrc, const uint32_t Wdst,
            const decx::bp::_cpu_transpose_config<1>* _config, decx::utils::_thread_arrange_1D* t1D);


        /**
        * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
        * @param Wsrc : width of source matrix (in element)
        * @param Wdst : height of destinated matrix (in element)
        */
        void transpose_4x4_caller(const float* src, float* dst, const uint32_t Wsrc, const uint32_t Wdst,
            const decx::bp::_cpu_transpose_config<4>* _config, decx::utils::_thread_arrange_1D* t1D);


        /**
        * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
        * @param Wsrc : width of source matrix (in element)
        * @param Wdst : height of destinated matrix (in element)
        */
        void transpose_2x2_caller(const double* src, double* dst, const uint32_t Wsrc, const uint32_t Wdst,
            const decx::bp::_cpu_transpose_config<8>* _config, decx::utils::_thread_arrange_1D* t1D);


        void transpose_MK_2x2_caller(const double* src, double* dst, const uint32_t Wsrc, const uint32_t Wdst,
            const decx::bp::_cpu_transpose_MK_config<8>* _config, decx::utils::_thread_arrange_1D* t1D);
    }
}


#ifdef __GNUC__
_THREAD_CALL_ static inline void
decx::bp::CPUK::block8x8_transpose_u8(__m64 _regs0[8], __m64 _regs1[8])
{
    const __m64 _mask_2x2_front = _mm_setr_pi16(0xFFFF, 0, 0xFFFF, 0);
    const __m64 _mask_2x2_back = _mm_setr_pi16(0, 0xFFFF, 0, 0xFFFF);
    const __m64 _mask_1x1_even = _mm_setr_pi8(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
    const __m64 _mask_1x1_odd = _mm_setr_pi8(0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF);

    // Transpose 4x4
    _regs1[0] = _mm_unpacklo_pi32(_regs0[0], _regs0[4]);
    _regs1[1] = _mm_unpacklo_pi32(_regs0[1], _regs0[5]);
    _regs1[2] = _mm_unpacklo_pi32(_regs0[2], _regs0[6]);
    _regs1[3] = _mm_unpacklo_pi32(_regs0[3], _regs0[7]);

    _regs1[4] = _mm_unpackhi_pi32(_regs0[0], _regs0[4]);
    _regs1[5] = _mm_unpackhi_pi32(_regs0[1], _regs0[5]);
    _regs1[6] = _mm_unpackhi_pi32(_regs0[2], _regs0[6]);
    _regs1[7] = _mm_unpackhi_pi32(_regs0[3], _regs0[7]);

    // Transpose 2x2
    _regs0[0] = _mm_xor_si64(_mm_and_si64(_regs1[0], _mask_2x2_front), 
                             _mm_and_si64(_mm_slli_si64(_regs1[2], 16), _mask_2x2_back));
    _regs0[1] = _mm_xor_si64(_mm_and_si64(_regs1[1], _mask_2x2_front), 
                             _mm_and_si64(_mm_slli_si64(_regs1[3], 16), _mask_2x2_back));

    _regs0[2] = _mm_xor_si64(_mm_and_si64(_regs1[2], _mask_2x2_back), 
                             _mm_and_si64(_mm_srli_si64(_regs1[0], 16), _mask_2x2_front));
    _regs0[3] = _mm_xor_si64(_mm_and_si64(_regs1[3], _mask_2x2_back), 
                             _mm_and_si64(_mm_srli_si64(_regs1[1], 16), _mask_2x2_front));

    _regs0[4] = _mm_xor_si64(_mm_and_si64(_regs1[4], _mask_2x2_front), 
                             _mm_and_si64(_mm_slli_si64(_regs1[6], 16), _mask_2x2_back));
    _regs0[5] = _mm_xor_si64(_mm_and_si64(_regs1[5], _mask_2x2_front), 
                             _mm_and_si64(_mm_slli_si64(_regs1[7], 16), _mask_2x2_back));

    _regs0[6] = _mm_xor_si64(_mm_and_si64(_regs1[6], _mask_2x2_back), 
                             _mm_and_si64(_mm_srli_si64(_regs1[4], 16), _mask_2x2_front));
    _regs0[7] = _mm_xor_si64(_mm_and_si64(_regs1[7], _mask_2x2_back), 
                             _mm_and_si64(_mm_srli_si64(_regs1[5], 16), _mask_2x2_front));

    // Transpose 1x1
    _regs1[0] = _mm_xor_si64(_mm_and_si64(_regs0[0], _mask_1x1_even), 
                             _mm_and_si64(_mm_slli_si64(_regs0[1], 8), _mask_1x1_odd));
    _regs1[1] = _mm_xor_si64(_mm_and_si64(_regs0[1], _mask_1x1_odd), 
                             _mm_and_si64(_mm_srli_si64(_regs0[0], 8), _mask_1x1_even));

    _regs1[2] = _mm_xor_si64(_mm_and_si64(_regs0[2], _mask_1x1_even), 
                             _mm_and_si64(_mm_slli_si64(_regs0[3], 8), _mask_1x1_odd));
    _regs1[3] = _mm_xor_si64(_mm_and_si64(_regs0[3], _mask_1x1_odd), 
                             _mm_and_si64(_mm_srli_si64(_regs0[2], 8), _mask_1x1_even));

    _regs1[4] = _mm_xor_si64(_mm_and_si64(_regs0[4], _mask_1x1_even), 
                             _mm_and_si64(_mm_slli_si64(_regs0[5], 8), _mask_1x1_odd));
    _regs1[5] = _mm_xor_si64(_mm_and_si64(_regs0[5], _mask_1x1_odd), 
                             _mm_and_si64(_mm_srli_si64(_regs0[4], 8), _mask_1x1_even));

    _regs1[6] = _mm_xor_si64(_mm_and_si64(_regs0[6], _mask_1x1_even), 
                             _mm_and_si64(_mm_slli_si64(_regs0[7], 8), _mask_1x1_odd));
    _regs1[7] = _mm_xor_si64(_mm_and_si64(_regs0[7], _mask_1x1_odd), 
                             _mm_and_si64(_mm_srli_si64(_regs0[6], 8), _mask_1x1_even));
}
#endif

#ifdef _MSC_VER
_THREAD_CALL_ static inline void
decx::bp::CPUK::block8x8_transpose_u8(uint64_t _regs0[8], uint64_t _regs1[8])
{
    const uint64_t _mask_2x2_front = 0xFFFF0000FFFF0000;
    const uint64_t _mask_2x2_back = 0x0000FFFF0000FFFF;
    const uint64_t _mask_1x1_even = 0xFF00FF00FF00FF00;
    const uint64_t _mask_1x1_odd = 0x00FF00FF00FF00FF;

    // Transpose 4x4
    _regs1[0] = (_regs0[0] & 0x00000000FFFFFFFF) ^ ((_regs0[4] << 32) & 0xFFFFFFFF00000000);
    _regs1[1] = (_regs0[1] & 0x00000000FFFFFFFF) ^ ((_regs0[5] << 32) & 0xFFFFFFFF00000000);
    _regs1[2] = (_regs0[2] & 0x00000000FFFFFFFF) ^ ((_regs0[6] << 32) & 0xFFFFFFFF00000000);
    _regs1[3] = (_regs0[3] & 0x00000000FFFFFFFF) ^ ((_regs0[7] << 32) & 0xFFFFFFFF00000000);

    _regs1[4] = (_regs0[4] & 0xFFFFFFFF00000000) ^ ((_regs0[0] >> 32) & 0x00000000FFFFFFFF);
    _regs1[5] = (_regs0[5] & 0xFFFFFFFF00000000) ^ ((_regs0[1] >> 32) & 0x00000000FFFFFFFF);
    _regs1[6] = (_regs0[6] & 0xFFFFFFFF00000000) ^ ((_regs0[2] >> 32) & 0x00000000FFFFFFFF);
    _regs1[7] = (_regs0[7] & 0xFFFFFFFF00000000) ^ ((_regs0[3] >> 32) & 0x00000000FFFFFFFF);

    // Transpose 2x2
    _regs0[0] = (_regs1[0] & 0x0000FFFF0000FFFF) ^ ((_regs1[2] << 16) & 0xFFFF0000FFFF0000);
    _regs0[1] = (_regs1[1] & 0x0000FFFF0000FFFF) ^ ((_regs1[3] << 16) & 0xFFFF0000FFFF0000);
    _regs0[2] = (_regs1[2] & 0xFFFF0000FFFF0000) ^ ((_regs1[0] >> 16) & 0x0000FFFF0000FFFF);
    _regs0[3] = (_regs1[3] & 0xFFFF0000FFFF0000) ^ ((_regs1[1] >> 16) & 0x0000FFFF0000FFFF);

    _regs0[4] = (_regs1[4] & 0x0000FFFF0000FFFF) ^ ((_regs1[6] << 16) & 0xFFFF0000FFFF0000);
    _regs0[5] = (_regs1[5] & 0x0000FFFF0000FFFF) ^ ((_regs1[7] << 16) & 0xFFFF0000FFFF0000);
    _regs0[6] = (_regs1[6] & 0xFFFF0000FFFF0000) ^ ((_regs1[4] >> 16) & 0x0000FFFF0000FFFF);
    _regs0[7] = (_regs1[7] & 0xFFFF0000FFFF0000) ^ ((_regs1[5] >> 16) & 0x0000FFFF0000FFFF);

    // Transpose 1x1
    _regs1[0] = (_regs0[0] & 0x00FF00FF00FF00FF) ^ ((_regs0[1] << 8) & 0xFF00FF00FF00FF00);
    _regs1[1] = (_regs0[1] & 0xFF00FF00FF00FF00) ^ ((_regs0[0] >> 8) & 0x00FF00FF00FF00FF);

    _regs1[2] = (_regs0[2] & 0x00FF00FF00FF00FF) ^ ((_regs0[3] << 8) & 0xFF00FF00FF00FF00);
    _regs1[3] = (_regs0[3] & 0xFF00FF00FF00FF00) ^ ((_regs0[2] >> 8) & 0x00FF00FF00FF00FF);

    _regs1[4] = (_regs0[4] & 0x00FF00FF00FF00FF) ^ ((_regs0[5] << 8) & 0xFF00FF00FF00FF00);
    _regs1[5] = (_regs0[5] & 0xFF00FF00FF00FF00) ^ ((_regs0[4] >> 8) & 0x00FF00FF00FF00FF);

    _regs1[6] = (_regs0[6] & 0x00FF00FF00FF00FF) ^ ((_regs0[7] << 8) & 0xFF00FF00FF00FF00);
    _regs1[7] = (_regs0[7] & 0xFF00FF00FF00FF00) ^ ((_regs0[6] >> 8) & 0x00FF00FF00FF00FF);

}
#endif


#define _AVX_MM128_TRANSPOSE_4X4_(src4, reg4) {             \
    reg4[0] = _mm_shuffle_ps(src4[0], src4[1], 0x44);       \
    reg4[2] = _mm_shuffle_ps(src4[0], src4[1], 0xEE);       \
    reg4[1] = _mm_shuffle_ps(src4[2], src4[3], 0x44);       \
    reg4[3] = _mm_shuffle_ps(src4[2], src4[3], 0xEE);       \
                                                            \
    src4[0] = _mm_shuffle_ps(reg4[0], reg4[1], 0x88);       \
    src4[1] = _mm_shuffle_ps(reg4[0], reg4[1], 0xDD);       \
    src4[2] = _mm_shuffle_ps(reg4[2], reg4[3], 0x88);       \
    src4[3] = _mm_shuffle_ps(reg4[2], reg4[3], 0xDD);       \
}


#define _AVX_MM128_TRANSPOSE_2X2_(src2, dst2) {             \
    dst2[0] = _mm_shuffle_pd(src2[0], src2[1], 0);          \
    dst2[1] = _mm_shuffle_pd(src2[0], src2[1], 3);          \
}


#define _AVX_MM256_TRANSPOSE_2X2_(src2, dst2) {                 \
    dst2[0] = _mm256_permute2f128_pd(src2[0], src2[1], 0x20);   \
    dst2[1] = _mm256_permute2f128_pd(src2[0], src2[1], 0x31);   \
}                                                               \



template <uint8_t _element_byte>
struct decx::bp::_cpu_transpose_config
{
    decx::utils::frag_manager _f_mgr;
    uint2 _src_proc_dims;

    _cpu_transpose_config() {}


    _cpu_transpose_config(const uint2 _proc_dims_src, 
                          const uint32_t _thread_num) 
    {
        uint8_t _alignment = 16 / _element_byte;
        if (_element_byte == 1) {
            _alignment = 8;
        }

        this->_src_proc_dims = _proc_dims_src;
        decx::utils::frag_manager_gen_Nx(&this->_f_mgr, _proc_dims_src.y, _thread_num, _alignment);
    }


    template <uint8_t _byte_src>
    decx::bp::_cpu_transpose_config<_element_byte>& operator=(const decx::bp::_cpu_transpose_config<_byte_src>& src)
    {
        if ((void*)this != (void*)(&src)) {
            this->_f_mgr = src._f_mgr;
            this->_src_proc_dims = src._src_proc_dims;
        }
        return *this;
    }
};



template <uint8_t _element_byte>
struct decx::bp::_cpu_transpose_MK_config
{
    decx::utils::frag_manager _f_mgr;
    uint2 _src_proc_dims;
    uint64_t _gapsrc_v1, _gapdst_v1;
    uint32_t _channel_num;


    _cpu_transpose_MK_config() {}


    _cpu_transpose_MK_config(const uint2 _proc_dims_src, 
                             const uint32_t _thread_num, 
                             const uint32_t _channel_num,
                             const uint64_t _gapsrc_v1, 
                             const uint64_t _gapdst_v1) 
    {
        uint8_t _alignment = 16 / _element_byte;
        this->_channel_num = _channel_num;
        this->_gapsrc_v1 = _gapsrc_v1;
        this->_gapdst_v1 = _gapdst_v1;

        this->_src_proc_dims = _proc_dims_src;

        if (_element_byte == 1) {
            _alignment = 8;
        }
        if (_channel_num < _thread_num) {
            decx::utils::frag_manager_gen_Nx(&this->_f_mgr, _proc_dims_src.y, _thread_num, _alignment);
        }
        else {
            decx::utils::frag_manager_gen(&this->_f_mgr, this->_channel_num, _thread_num);
        }
    }

    template <uint8_t _byte_src>
    decx::bp::_cpu_transpose_MK_config<_element_byte>& operator=(const decx::bp::_cpu_transpose_MK_config<_byte_src>& src)
    {
        if ((void*)this != (void*)(&src)) {
            this->_f_mgr = src._f_mgr;
            this->_src_proc_dims = src._src_proc_dims;
            this->_gapdst_v1 = src._gapdst_v1;
            this->_gapsrc_v1 = src._gapsrc_v1;
            this->_channel_num = src._channel_num;
        }
        return *this;
    }
};


#endif
