/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _DEV_CONV2_BORDER_CONST_H_
#define _DEV_CONV2_BORDER_CONST_H_

#include "../../../../core/basic.h"
#include "sconv2_kernel_callers.h"
#include "conv2_border_const_fp32.h"
#include "../../../../classes/GPU_Matrix.h"
#include "../../../conv_utils.h"



namespace decx
{
    namespace conv 
    {
        template <bool _print>
        /*\
        * 8 x 8
        * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
        * In order to save the device memory, just allocate memory of suitable size
        */
        static void _dev_conv2_fp32_BC_R8x8(decx::conv::_cuda_conv2_fp32_preset* _conv2_preset, de::DH* handle);

        template <bool _print>
        /*\
        * 16 x 16
        * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
        * In order to save the device memory, just allocate memory of suitable size
        */
        static void _dev_conv2_fp32_BC_R16x16(decx::conv::_cuda_conv2_fp32_preset* _conv2_preset, de::DH* handle);

        template <bool _print>
        /*\
        * 8 x 16
        * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
        * In order to save the device memory, just allocate memory of suitable size
        */
        static void _dev_conv2_fp32_BC_R8x16(decx::conv::_cuda_conv2_fp32_preset* _conv2_preset, de::DH* handle);

        template <bool _print>
        /*\
        * 16 x 8
        * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
        * In order to save the device memory, just allocate memory of suitable size
        */
        static void _dev_conv2_fp32_BC_R16x8(decx::conv::_cuda_conv2_fp32_preset* _conv2_preset, de::DH* handle);


        template <bool _print>
        static void dev_conv2_fp32_border_zero(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst, de::DH* handle);
    }
}



template <bool _print> static void
decx::conv::_dev_conv2_fp32_BC_R8x8(decx::conv::_cuda_conv2_fp32_preset* _conv2_preset, de::DH* handle)
{
    decx::conv::_conv2_fp32_BC_pre_process<_print, bounded_kernel_R8, bounded_kernel_R8>(_conv2_preset, handle);

    decx::conv::conv2_fp32_kernel_r8x8((float4*)_conv2_preset->src_buf.ptr, (float*)_conv2_preset->ker_buf.ptr, 
        (float4*)_conv2_preset->_Kparams._dst_confs._ptr,
        make_uint2(_conv2_preset->_Kparams.kernel_shift.x, _conv2_preset->_Kparams.kernel_shift.y),
        make_uint2(_conv2_preset->_Kparams.src_buf_dims.x / 4, _conv2_preset->_Kparams.src_buf_dims.y),
        make_uint2(_conv2_preset->_Kparams.dst_dims.x / 4, _conv2_preset->_Kparams.dst_dims.y),
        _conv2_preset->_Kparams.ker_dims, _conv2_preset->S);

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}




template <bool _print> static void
decx::conv::_dev_conv2_fp32_BC_R16x16(decx::conv::_cuda_conv2_fp32_preset* _conv2_preset, de::DH* handle)
{
    decx::conv::_conv2_fp32_BC_pre_process<_print, bounded_kernel_R16, bounded_kernel_R16>(_conv2_preset, handle);

    decx::conv::conv2_fp32_kernel_r16x16((float4*)_conv2_preset->src_buf.ptr, (float*)_conv2_preset->ker_buf.ptr, 
        (float4*)_conv2_preset->_Kparams._dst_confs._ptr,
        make_uint2(_conv2_preset->_Kparams.kernel_shift.x, _conv2_preset->_Kparams.kernel_shift.y),
        make_uint2(_conv2_preset->_Kparams.src_buf_dims.x / 4, _conv2_preset->_Kparams.src_buf_dims.y),
        make_uint2(_conv2_preset->_Kparams.dst_dims.x / 4, _conv2_preset->_Kparams.dst_dims.y),
        _conv2_preset->_Kparams.ker_dims, _conv2_preset->S);

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}




template <bool _print> static void
decx::conv::_dev_conv2_fp32_BC_R8x16(decx::conv::_cuda_conv2_fp32_preset* _conv2_preset, de::DH* handle)
{
    decx::conv::_conv2_fp32_BC_pre_process<_print, bounded_kernel_R8, bounded_kernel_R16>(_conv2_preset, handle);

    decx::conv::conv2_fp32_kernel_r8x16((float4*)_conv2_preset->src_buf.ptr, (float*)_conv2_preset->ker_buf.ptr, 
        (float4*)_conv2_preset->_Kparams._dst_confs._ptr,
        make_uint2(_conv2_preset->_Kparams.kernel_shift.x, _conv2_preset->_Kparams.kernel_shift.y),
        make_uint2(_conv2_preset->_Kparams.src_buf_dims.x / 4, _conv2_preset->_Kparams.src_buf_dims.y),
        make_uint2(_conv2_preset->_Kparams.dst_dims.x / 4, _conv2_preset->_Kparams.dst_dims.y),
        _conv2_preset->_Kparams.ker_dims, _conv2_preset->S);

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}




template <bool _print> static void
decx::conv::_dev_conv2_fp32_BC_R16x8(decx::conv::_cuda_conv2_fp32_preset* _conv2_preset, de::DH* handle)
{
    decx::conv::_conv2_fp32_BC_pre_process<_print, bounded_kernel_R16, bounded_kernel_R8>(_conv2_preset, handle);

    decx::conv::conv2_fp32_kernel_r16x8((float4*)_conv2_preset->src_buf.ptr, (float*)_conv2_preset->ker_buf.ptr, 
        (float4*)_conv2_preset->_Kparams._dst_confs._ptr,
        make_uint2(_conv2_preset->_Kparams.kernel_shift.x,        _conv2_preset->_Kparams.kernel_shift.y),
        make_uint2(_conv2_preset->_Kparams.src_buf_dims.x / 4,    _conv2_preset->_Kparams.src_buf_dims.y),
        make_uint2(_conv2_preset->_Kparams.dst_dims.x / 4,        _conv2_preset->_Kparams.dst_dims.y),
        _conv2_preset->_Kparams.ker_dims, _conv2_preset->S);

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}




template <bool _print> static void
decx::conv::dev_conv2_fp32_border_zero(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst, de::DH* handle)
{
    int2 half_ker_dim;
    half_ker_dim.x = kernel->Width() / 2;                half_ker_dim.y = kernel->Height() / 2;

    dst->re_construct(src->Type(), src->Width(), src->Height());

    decx::conv::_cuda_conv2_fp32_preset _conv2_preset;
    _conv2_preset.memcpy_flag = cudaMemcpyDeviceToDevice;
    _conv2_preset._Kparams._src_confs.gen_matrix_configs(src);
    _conv2_preset._Kparams._kernel_confs.gen_matrix_configs(kernel);
    _conv2_preset._Kparams._dst_confs.gen_matrix_configs(dst);

    if (half_ker_dim.x < bounded_kernel_R8 + 1) {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::conv::_dev_conv2_fp32_BC_R8x8<_print>(&_conv2_preset, handle);
        }
        else {
            decx::conv::_dev_conv2_fp32_BC_R16x8<_print>(&_conv2_preset, handle);
        }
    }
    else {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::conv::_dev_conv2_fp32_BC_R8x16<_print>(&_conv2_preset, handle);
        }
        else {
            decx::conv::_dev_conv2_fp32_BC_R16x16<_print>(&_conv2_preset,  handle);
        }
    }

    _conv2_preset.release();
}


#endif        //    #ifndef _DEV_CONV2_BORDER_IGNORED_H_