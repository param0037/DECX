/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "conv2_im2row_fp32.h"



void decx::conv_I2R::conv2_im2row_fp32_NB_stride(decx::_Tensor* src, decx::_TensorArray* kernel, decx::_Tensor* dst, const uint2 strideXY, de::DH* handle)
{
    const uint2 dst_dims = make_uint2((src->width - (kernel->width - 1)) / strideXY.x, (src->height - (kernel->height - 1)) / strideXY.y);
    // reshape destinated tensor
    dst->re_construct(decx::_DATA_TYPES_FLAGS_::_FP32_, dst_dims.x, dst_dims.y, kernel->tensor_num, decx::DATA_STORE_TYPE::Page_Default);

    const uint2 ker_dims = make_uint2(kernel->width, kernel->height);

    // dimensions of im2col_buffer
    const uint dst_wpitch = decx::utils::ceil<uint>(dst_dims.x, 4) * 4;
    const ulong2 I2C_buf_dim = make_ulong2(kernel->wpitch * kernel->height * kernel->dpitch, dst_wpitch * dst_dims.y);

    // How many times needed to go through the whole proccedure
    const uint _times = (uint)decx::utils::ceil<size_t>(I2C_buf_dim.x * I2C_buf_dim.y, _I2C_frag_size_CPU_);

    // apply the times as division on the height of source matrix
    decx::utils::frag_manager f_mgr_dstH;
    decx::utils::frag_manager_gen(&f_mgr_dstH, dst_dims.y, _times);

    const size_t I2C_alloc_H = dst_wpitch * max(f_mgr_dstH.frag_len, f_mgr_dstH.frag_left_over);

    // allocate buffer for I2C_buf_frag
    decx::PtrInfo<float> I2C_buf, row_buf, arranged_kernel;
    if (decx::alloc::_host_virtual_page_malloc(&I2C_buf, I2C_buf_dim.x * I2C_alloc_H * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&row_buf, decx::cpI.cpu_concurrency * kernel->dpitch * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }

    decx::utils::_thread_arrange_1D t1D(decx::cpI.cpu_concurrency);

    const float* loc_src = (float*)src->Tens.ptr;
    float* loc_dst = (float*)dst->Tens.ptr;

    const size_t src_frag = src->dp_x_wp * f_mgr_dstH.frag_len * strideXY.y,
        dst_frag = dst->dpitch * f_mgr_dstH.frag_len * dst_wpitch;

    // start the loop
    for (int i = 0; i < f_mgr_dstH.frag_num - 1; ++i) {
        decx::conv_I2R::_im2row_caller_fp32_stride(loc_src, row_buf.ptr, I2C_buf.ptr, strideXY, kernel->dpitch / 4, ker_dims,
            kernel->wpitch, src->dp_x_wp, I2C_buf_dim.x, make_uint2(dst_wpitch, f_mgr_dstH.frag_len), &t1D);

        decx::conv_I2R::_im2row_eq_GEMM_caller_fp32(I2C_buf.ptr, (float*)kernel->TensArr.ptr, loc_dst,
            I2C_buf_dim.x, dst_wpitch * f_mgr_dstH.frag_len, dst->dpitch, kernel->tensor_num, &t1D);

        loc_src += src_frag;
        loc_dst += dst_frag;
    }
    // the remaining part of the loop
    const uint _L = f_mgr_dstH.is_left ? f_mgr_dstH.frag_left_over : f_mgr_dstH.frag_len;

    decx::conv_I2R::_im2row_caller_fp32_stride(loc_src, row_buf.ptr, I2C_buf.ptr, strideXY, kernel->dpitch / 4, ker_dims,
        kernel->wpitch, src->dp_x_wp, I2C_buf_dim.x, make_uint2(dst_wpitch, _L), &t1D);

    decx::conv_I2R::_im2row_eq_GEMM_caller_fp32(I2C_buf.ptr, (float*)kernel->TensArr.ptr, loc_dst,
        I2C_buf_dim.x, dst_wpitch * _L, dst->dpitch, kernel->tensor_num, &t1D);

    decx::alloc::_host_virtual_page_dealloc(&I2C_buf);
    decx::alloc::_host_virtual_page_dealloc(&row_buf);
}



void decx::conv_I2R::conv2_im2row_fp32_NB(decx::_Tensor* src, decx::_TensorArray* kernel, decx::_Tensor* dst, de::DH* handle)
{
    const uint2 dst_dims = make_uint2(src->width - (kernel->width - 1), src->height - (kernel->height - 1));
    // reshape destinated tensor
    dst->re_construct(src->type, dst_dims.x, dst_dims.y, kernel->tensor_num, decx::DATA_STORE_TYPE::Page_Default);

    const uint2 ker_dims = make_uint2(kernel->width, kernel->height);

    // dimensions of im2col_buffer
    const uint dst_wpitch = decx::utils::ceil<uint>(dst_dims.x, 4) * 4;
    const ulong2 I2C_buf_dim = make_ulong2(kernel->wpitch * kernel->height * kernel->dpitch, dst_wpitch * dst_dims.y);

    // How many times needed to go through the whole proccedure
    const uint _times = (uint)decx::utils::ceil<size_t>(I2C_buf_dim.x * I2C_buf_dim.y, _I2C_frag_size_CPU_);

    // apply the times as division on the height of source matrix
    decx::utils::frag_manager f_mgr_dstH;
    decx::utils::frag_manager_gen(&f_mgr_dstH, dst_dims.y, _times);

    const size_t I2C_alloc_H = dst_wpitch * max(f_mgr_dstH.frag_len, f_mgr_dstH.frag_left_over);

    // allocate buffer for I2C_buf_frag
    decx::PtrInfo<float> I2C_buf, row_buf, arranged_kernel;
    if (decx::alloc::_host_virtual_page_malloc(&I2C_buf, I2C_buf_dim.x * I2C_alloc_H * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&row_buf, decx::cpI.cpu_concurrency * kernel->dpitch * kernel->width * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }

    decx::utils::_thread_arrange_1D t1D(decx::cpI.cpu_concurrency);

    const float* loc_src = (float*)src->Tens.ptr;
    float* loc_dst = (float*)dst->Tens.ptr;

    const size_t src_frag = src->dp_x_wp * f_mgr_dstH.frag_len,
        dst_frag = dst->dpitch * f_mgr_dstH.frag_len * dst_wpitch;

    // start the loop
    for (int i = 0; i < f_mgr_dstH.frag_num - 1; ++i) {
        decx::conv_I2R::_im2row_caller_fp32(loc_src, row_buf.ptr, I2C_buf.ptr, kernel->dpitch / 4, ker_dims,
            kernel->wpitch, src->dp_x_wp, I2C_buf_dim.x, make_uint2(dst_wpitch, f_mgr_dstH.frag_len), &t1D);

        decx::conv_I2R::_im2row_eq_GEMM_caller_fp32(I2C_buf.ptr, (float*)kernel->TensArr.ptr, loc_dst,
            I2C_buf_dim.x, dst_wpitch * f_mgr_dstH.frag_len, dst->dpitch, kernel->tensor_num, &t1D);

        loc_src += src_frag;
        loc_dst += dst_frag;
    }
    // the remaining part of the loop
    const uint _L = f_mgr_dstH.is_left ? f_mgr_dstH.frag_left_over : f_mgr_dstH.frag_len;

    decx::conv_I2R::_im2row_caller_fp32(loc_src, row_buf.ptr, I2C_buf.ptr, kernel->dpitch / 4, ker_dims,
        kernel->wpitch, src->dp_x_wp, I2C_buf_dim.x, make_uint2(dst_wpitch, _L), &t1D);

    decx::conv_I2R::_im2row_eq_GEMM_caller_fp32(I2C_buf.ptr, (float*)kernel->TensArr.ptr, loc_dst,
        I2C_buf_dim.x, dst_wpitch * _L, dst->dpitch, kernel->tensor_num, &t1D);

    decx::alloc::_host_virtual_page_dealloc(&I2C_buf);
    decx::alloc::_host_virtual_page_dealloc(&row_buf);
}
