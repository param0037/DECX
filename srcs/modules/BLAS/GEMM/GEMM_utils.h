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


#ifndef _GEMM_UTILS_H_
#define _GEMM_UTILS_H_

#include "../../core/basic.h"
#include "../../classes/Matrix.h"
#include "../../core/thread_management/thread_arrange.h"
#include "../../core/utils/fragment_arrangment.h"
#include "../../core/resources_manager/decx_resource.h"


#define GEMM_BlockDim 16

namespace de
{
    enum GEMM_properties
    {
        HALF_GEMM_DIRECT    = 0,
        HALF_GEMM_ACCURATE  = 1
    };
}


#ifdef _DECX_BLAS_CPU_
//namespace decx
//{
//    template <typename _data_type>
//    class cpu_GEMM_general_config
//    {
//    private:
//        const decx::_matrix_layout* _layout_A, *_layout_B, *_layout_C;
//        uint32_t _concurrency;
//
//        uint2 _thread_dist;
//        decx::utils::_thread_arrange_2D _t2D;
//        decx::utils::frag_manager _f_mgr_sort_B, _pre_f_mgrW;
//        uint32_t _L;
//        uint2 _arranged_B_dims;
//        uint32_t _pitch_extdst;
//
//        decx::PtrInfo<void> _arranged_B, _extra_dst;
//
//    public:
//        cpu_GEMM_general_config() {
//            memset(this, 0, sizeof(cpu_GEMM_general_config<_data_type>));
//        }
//
//
//        _CRSR_ void Config(const uint32_t parallel, const decx::_matrix_layout* layout_A, 
//            const decx::_matrix_layout* layout_B, de::DH *handle);
//
//
//        bool Changed(const uint32_t parallel, const decx::_matrix_layout* layout_A,
//            const decx::_matrix_layout* layout_B) const;
//
//
//        _CRSR_ static void Validate(de::DH* handle, const decx::_matrix_layout* layout_A,
//            const decx::_matrix_layout* layout_B, const decx::_matrix_layout* layout_C = NULL);
//
//
//        template <bool _is_cpl>
//        void Run(decx::_Matrix* _A, decx::_Matrix* _B, decx::_Matrix* _dst);
//
//        template <bool _is_cpl>
//        void Run(decx::_Matrix* _A, decx::_Matrix* _B, decx::_Matrix* _C, decx::_Matrix* _dst);
//
//
//        static void release_buffers(decx::cpu_GEMM_general_config<_data_type>* _fake_this);
//    };
//
//    extern decx::ResourceHandle _cpu_GEMM_fp32_mgr;
//    extern decx::ResourceHandle _cpu_GEMM_b64_mgr;
//
//
//    typedef struct GEMM_AB_configs
//    {
//        uint _pitchA, _pitchB, _pitchdst, _pitchC, _linear;
//        uint2 _proc_dims;
//
//        GEMM_AB_configs() {}
//
//
//        GEMM_AB_configs(const uint32_t pitchA, 
//                        const uint32_t pitchB, 
//                        const uint32_t pitchdst, 
//                        const uint32_t linear, 
//                        const uint2 proc_dims, 
//                        const uint32_t pitchC = 0) :
//            _pitchA(pitchA),
//            _pitchB(pitchB),
//            _pitchdst(pitchdst),
//            _linear(linear),
//            _pitchC(pitchC),
//            _proc_dims(proc_dims) {}
//
//    }_C_MM_;
//}
//
//
//namespace decx
//{
//    namespace gemm {
//        namespace CPUK {
//            /*
//            * @param Wsrc : width of src, in float
//            * @param Wdst : width of dst, in float
//            * @param cpy_dim : ~.x -> width to copy (in __m256); ~.y -> height to copy (in float)
//            */
//            void GEMM_fp32_cpy_L8(const float* src, float* dst, const uint Wsrc, const uint Wdst, const uint2 cpy_dim);
//
//
//            /*
//            * @param Wsrc : width of src, in float
//            * @param Wdst : width of dst, in float
//            * @param cpy_dim : ~.x -> width to copy (in __m256); ~.y -> height to copy (in float)
//            */
//            void GEMM_fp64_cpy_L8(const double* src, double* dst, const uint Wsrc, const uint Wdst, const uint2 cpy_dim);
//        }
//    }
//}



#endif

#endif