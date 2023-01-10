/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CONV_UTILS_H_
#define _CONV_UTILS_H_

#ifdef _DECX_CPU_CODES_
#include "../core/vector_defines.h"
#include "../core/utils/fragment_arrangment.h"
#include "../core/thread_management/thread_arrange.h"
#endif


#include "../classes/Matrix.h"
#include "../classes/MatrixArray.h"
#include "../classes/Tensor.h"
#include "../classes/TensorArray.h"

#ifdef _DECX_CUDA_CODES_
#include "../classes/GPU_Matrix.h"
#include "../classes/GPU_MatrixArray.h"
#include "../classes/GPU_Tensor.h"
#include "../classes/GPU_TensorArray.h"
#endif

#ifdef _DECX_CPU_CODES_
#include "../basic_process/rect_and_cube/CPU/rect_copy2D_exec.h"
#endif


namespace decx {
    enum conv_property
    {
        de_conv_no_compensate = 0,
        de_conv_zero_compensate = 1,

        half_conv_ordinary = 2,
        half_conv_accurate = 3
    };
}



//namespace decx
//{
//    /**
//    * @param __x : the destinated matrix
//    * @param dst_dim .x -> width; .y -> height; .z -> _store_type
//    */
//    static void conv2_dst_rearrangement(_Matrix* __x, const uint3 dst_dim) {
//        __x->re_construct(__x->type, dst_dim.x, dst_dim.y, dst_dim.z);
//    }
//
//
//    /**
//    * @param __x : the _MatrixArray wait to be rearranged
//    * @param dst_dim .x -> width; .y -> height; .z -> ArrayNum; .w -> _store_type
//    */
//    static void conv2_mc_dst_rearrangement(_MatrixArray* __x, uint4 dst_dim) {
//        __x->re_construct(__x->type, dst_dim.x, dst_dim.y, dst_dim.z, dst_dim.w);
//    }
//
//
//    /*template <typename T>
//    static void conv_tensor_rearrangement(decx::_Tensor<T>* __x, const uint4 dst_dim) {
//        __x->re_construct(dst_dim.x, dst_dim.y, dst_dim.z, dst_dim.w);
//    }*/
//}




#ifdef _DECX_CUDA_CODES_


#define N_conv_once 1024 * 1024

// the total bytes (IN BYTE) allowed for kernel to restore in constant memory
#define kernel_in_CM 512


#define conv2_bld 16
#define conv2_tps 4

#define bounded_kernel_R8  8
#define bounded_kernel_R16 16


//namespace decx
//{
//    /**
//    * @param __x : the destinated matrix
//    * @param dst_dim .x -> width; .y -> height; .z -> MatNum
//    */
//    template <typename T>
//    static void _dev_conv2_dst_rearrangement(_GPU_MatrixArray* __x, uint3 dst_dim) {
//        __x->re_construct(__x->type, dst_dim.x, dst_dim.y, dst_dim.z);
//    }
//
//
//    /**
//    * @param __x : the destinated matrix
//    * @param dst_dim .x -> width; .y -> height
//    */
//    template <typename T>
//    static void _dev_conv2_dst_rearrangement(_GPU_Matrix* __x, uint2 dst_dim) {
//        __x->re_construct(__x->type, dst_dim.x, dst_dim.y);
//    }
//}
#endif

#ifdef _DECX_CPU_CODES_
namespace decx
{
    typedef struct _Conv2_MK_Props_fp32
    {
        uint2 ker_dims;
        uint Wdst;
        uint Wsrc;

        ushort reg_WL;
        uint _loop;

        decx::utils::frag_manager* f_mgr; 
        size_t page_size_dst;
        size_t page_size_src;
        size_t page_size_ker;
        uint channel_size;


        _Conv2_MK_Props_fp32(const uint2                 _ker_dims, 
                             const uint                  _W_original_src, 
                             const uint                  _W_tmp_src, 
                             const size_t                _page_size, 
                             const uint                  _channel_size,
                             decx::utils::frag_manager*  _f_mgr,
                             const size_t                _page_size_src = 0,      // set only when non-border conv2d occurs
                             const size_t                _page_size_ker = 0       // set only when multi-kernel conv2d occurs
        ) :
            ker_dims(_ker_dims), 
            Wdst(_W_original_src),
            Wsrc(_W_tmp_src),
            page_size_dst(_page_size),
            page_size_src(_page_size_src),
            page_size_ker(_page_size_ker),
            channel_size(_channel_size),
            f_mgr(_f_mgr)
        {
            const uint half_kernel_w = ker_dims.x / 2;
            if (half_kernel_w < 5) {
                this->_loop = 0;
                this->reg_WL = 0;
            }
            else {
                this->_loop = (uint)decx::utils::clamp_min<int>((((int)this->ker_dims.x / 2) - 5) / 4, 0);
                this->reg_WL = (ushort)(this->ker_dims.x - 1 - 8 - _loop * 8);
            }
        }

    }_C2_MK32;


    typedef struct _Conv2_MK_Props_fp64
    {
        uint2 ker_dims;
        uint Wdst;
        uint Wsrc;

        ushort reg_WL;
        uint _loop;

        decx::utils::frag_manager* f_mgr; 
        size_t page_size_dst;
        size_t page_size_src;
        size_t page_size_ker;
        uint channel_size;

        _Conv2_MK_Props_fp64(const uint2                 _ker_dims,
                             const uint                  _W_original_src, 
                             const uint                  _W_tmp_src, 
                             const size_t                _page_size, 
                             const uint                  _channel_size,
                             decx::utils::frag_manager*  _f_mgr,
                             const size_t                _page_size_src = 0,      // set only when non-border conv2d occurs
                             const size_t                _page_size_ker = 0       // set only when multi-kernel conv2d occurs
        ) :
            ker_dims(_ker_dims), 
            Wdst(_W_original_src),
            Wsrc(_W_tmp_src),
            page_size_dst(_page_size),
            page_size_src(_page_size_src),
            page_size_ker(_page_size_ker),
            channel_size(_channel_size),
            f_mgr(_f_mgr)
        {
            const uint half_kernel_w = ker_dims.x / 2;
            if (half_kernel_w < 5) {
                this->_loop = 0;
                this->reg_WL = 0;
            }
            else {
                this->_loop = (uint)decx::utils::clamp_min<int>((((int)this->ker_dims.x / 2) - 3) / 2, 0);
                this->reg_WL = (ushort)(this->ker_dims.x - 1 - 4 - _loop * 4);
            }
        }

    }_C2_MK64;
}


namespace decx
{
    static void _thread_dispatch_for_conv2(decx::utils::frag_manager** f_mgr,
        const size_t tot, const uint thread_num, const uint N, const uint Wproc);


    static void _thread_dispatch_for_conv2_fp64(decx::utils::frag_manager** f_mgr,
        const size_t tot, const uint thread_num, const uint N, const uint Wproc);
}


#define _CONV2_THR_DIST_CRIT_R1R4_ 4096
#define _CONV2_THR_DIST_CRIT_R5R8_ 1024
#define _CONV2_THR_DIST_CRIT_R9R12_ 256

#define _CONV2_THR_DIST_CRIT_FP64_R1R4_ 2048
#define _CONV2_THR_DIST_CRIT_FP64_R5R8_ 512
#define _CONV2_THR_DIST_CRIT_FP64_R9R12_ 128


static void decx::_thread_dispatch_for_conv2(decx::utils::frag_manager** f_mgr, const size_t tot,
    const uint thread_num, const uint N, const uint Wproc)
{
    decx::utils::frag_manager* f_mgr_N = new decx::utils::frag_manager;
    decx::utils::frag_manager_gen_Nx(f_mgr_N, tot, thread_num, N);

    if (f_mgr_N->is_left) {
        size_t _exceeded = (size_t)(f_mgr_N->frag_left_over - f_mgr_N->frag_len) * (size_t)Wproc;
        if (_exceeded > _CONV2_THR_DIST_CRIT_R5R8_) {
            decx::utils::frag_manager_gen(f_mgr_N, tot, thread_num);
            *f_mgr = f_mgr_N;
            return;
        }
    }
    *f_mgr = f_mgr_N;
}



static void decx::_thread_dispatch_for_conv2_fp64(decx::utils::frag_manager** f_mgr, const size_t tot,
    const uint thread_num, const uint N, const uint Wproc)
{
    decx::utils::frag_manager* f_mgr_N = new decx::utils::frag_manager;
    decx::utils::frag_manager_gen_Nx(f_mgr_N, tot, thread_num, N);

    if (f_mgr_N->is_left) {
        size_t _exceeded = (size_t)(f_mgr_N->frag_left_over - f_mgr_N->frag_len) * (size_t)Wproc;
        if (_exceeded > _CONV2_THR_DIST_CRIT_FP64_R5R8_) {
            decx::utils::frag_manager_gen(f_mgr_N, tot, thread_num);
            *f_mgr = f_mgr_N;
            return;
        }
    }
    *f_mgr = f_mgr_N;
}
#endif


#endif