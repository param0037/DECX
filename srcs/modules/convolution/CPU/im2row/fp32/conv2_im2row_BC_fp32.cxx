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
#include "../../../../basic_process/rect_and_cube/CPU/rect_copy2D_exec.h"


void decx::conv_I2R::conv2_im2row_fp32_BC_stride(decx::_Tensor* src, decx::_TensorArray* kernel, decx::_Tensor* dst, const uint2 strideXY, de::DH* handle)
{
    const uint2 dst_dims = make_uint2(src->width / strideXY.x, src->height / strideXY.y);
    // required dimensions of src_buffer
    const uint2 src_buf_dims = make_uint2(decx::utils::ceil<uint>(src->width + kernel->width - 1, 4) * 4, 
        src->height + kernel->height - 1);
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
    const size_t src_buf_dp_x_wp = src_buf_dims.x * src->dpitch;

    // allocate buffer for I2C_buf_frag
    decx::PtrInfo<float> src_buf, I2C_buf, row_buf, arranged_kernel;
    if (decx::alloc::_host_virtual_page_malloc(&src_buf, src_buf_dp_x_wp * src_buf_dims.y * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
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

    decx::_cpy2D_plane((float*)src->Tens.ptr, 
        DECX_PTR_SHF_XY<float, float>(src_buf.ptr, kernel->height / 2, kernel->width / 2 * src->dpitch, src_buf_dp_x_wp),
        src->dp_x_wp, src_buf_dp_x_wp, make_uint2(src->width * src->dpitch, src->height));

    decx::utils::_thread_arrange_1D t1D(decx::cpI.cpu_concurrency);

    const float* loc_src = (float*)src_buf.ptr;
    float* loc_dst = (float*)dst->Tens.ptr;

    const size_t src_frag = src_buf_dp_x_wp * f_mgr_dstH.frag_len * strideXY.y,
        dst_frag = (size_t)dst->dpitch * (size_t)f_mgr_dstH.frag_len * (size_t)dst_wpitch;

    // start the loop
    for (int i = 0; i < f_mgr_dstH.frag_num - 1; ++i) {
        decx::conv_I2R::_im2row_caller_fp32_stride(loc_src,                              row_buf.ptr,                    
                                                   I2C_buf.ptr,                          strideXY,              
                                                   kernel->dpitch / 4,                   ker_dims, 
                                                   kernel->wpitch,                       src_buf_dp_x_wp,                
                                                   I2C_buf_dim.x,                        make_uint2(dst_wpitch, f_mgr_dstH.frag_len), &t1D);

        decx::conv_I2R::_im2row_eq_GEMM_caller_fp32(I2C_buf.ptr,                          (float*)kernel->TensArr.ptr,        
                                                    loc_dst,                              I2C_buf_dim.x,    
                                                    dst_wpitch * f_mgr_dstH.frag_len,     dst->dpitch, 
                                                    kernel->tensor_num,                   &t1D);

        loc_src += src_frag;
        loc_dst += dst_frag;
    }
    // the remaining part of the loop
    const uint _L = f_mgr_dstH.is_left ? f_mgr_dstH.frag_left_over : f_mgr_dstH.frag_len;

    decx::conv_I2R::_im2row_caller_fp32_stride(loc_src,                         row_buf.ptr, 
                                               I2C_buf.ptr,                     strideXY, 
                                               kernel->dpitch / 4,              ker_dims,
                                               kernel->wpitch,                  src_buf_dp_x_wp, 
                                               I2C_buf_dim.x,                   make_uint2(dst_wpitch, _L), &t1D);

    decx::conv_I2R::_im2row_eq_GEMM_caller_fp32(I2C_buf.ptr,                    (float*)kernel->TensArr.ptr, 
                                                loc_dst,                        I2C_buf_dim.x, 
                                                dst_wpitch * _L, dst->dpitch,   kernel->tensor_num,         &t1D);

    // deallocate the spaces
    decx::alloc::_host_virtual_page_dealloc(&I2C_buf);
    decx::alloc::_host_virtual_page_dealloc(&src_buf);
    decx::alloc::_host_virtual_page_dealloc(&row_buf);
}



void decx::conv_I2R::conv2_im2row_fp32_BC(decx::_Tensor* src, decx::_TensorArray* kernel, decx::_Tensor* dst, de::DH* handle)
{
    const uint2 dst_dims = make_uint2(src->width, src->height);
    // reshape destinated tensor
    dst->re_construct(src->type, dst_dims.x, dst_dims.y, kernel->tensor_num, decx::DATA_STORE_TYPE::Page_Default);

    const uint2 ker_dims = make_uint2(kernel->width, kernel->height);

    // dimensions of im2col_buffer
    const ulong2 I2C_buf_dim = make_ulong2(kernel->wpitch * kernel->height * kernel->dpitch, src->wpitch * dst_dims.y);

    // How many times needed to go through the whole proccedure
    const uint _times = (uint)decx::utils::ceil<size_t>(I2C_buf_dim.x * I2C_buf_dim.y, _I2C_frag_size_CPU_);

    const uint2 src_buf_dims = make_uint2(decx::utils::ceil<uint>(src->width + kernel->width - 1, 4) * 4,
        src->height + kernel->height - 1);

    // apply the times as division on the height of source matrix
    decx::utils::frag_manager f_mgr_dstH;
    decx::utils::frag_manager_gen(&f_mgr_dstH, dst_dims.y, _times);

    const size_t I2C_alloc_H = src->wpitch * max(f_mgr_dstH.frag_len, f_mgr_dstH.frag_left_over);
    const size_t src_buf_dp_x_wp = (size_t)src_buf_dims.x * (size_t)src->dpitch;

    // allocate buffer for I2C_buf_frag
    decx::PtrInfo<float> I2C_buf, row_buf, src_buf;

    if (decx::alloc::_host_virtual_page_malloc(&src_buf, src_buf_dp_x_wp * src_buf_dims.y * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
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

    decx::_cpy2D_plane((float*)src->Tens.ptr,
        DECX_PTR_SHF_XY<float, float>(src_buf.ptr, kernel->height / 2, kernel->width / 2 * src->dpitch, src_buf_dp_x_wp),
        src->dp_x_wp, src_buf_dp_x_wp, make_uint2(src->width * src->dpitch, src->height));
    
    decx::utils::_thread_arrange_1D t1D(decx::cpI.cpu_concurrency);

    const float* loc_src = (float*)src_buf.ptr;
    float* loc_dst = (float*)dst->Tens.ptr;

    const size_t src_frag = src_buf_dp_x_wp * (size_t)f_mgr_dstH.frag_len,
        dst_frag = (size_t)dst->dpitch * (size_t)f_mgr_dstH.frag_len * (size_t)src->wpitch;

    // start the loop
    for (int i = 0; i < f_mgr_dstH.frag_num - 1; ++i) {
        decx::conv_I2R::_im2row_caller_fp32(loc_src,                                    row_buf.ptr, 
                                            I2C_buf.ptr,                                kernel->dpitch / 4, ker_dims,
                                            kernel->wpitch,                             src_buf_dp_x_wp, 
                                            I2C_buf_dim.x,                              make_uint2(src->wpitch, f_mgr_dstH.frag_len), &t1D);

        decx::conv_I2R::_im2row_eq_GEMM_caller_fp32(I2C_buf.ptr,                        (float*)kernel->TensArr.ptr, 
                                                    loc_dst,                            I2C_buf_dim.x, 
                                                    src->wpitch * f_mgr_dstH.frag_len,  dst->dpitch, 
                                                    kernel->tensor_num,                 &t1D);

        loc_src += src_frag;
        loc_dst += dst_frag;
    }
    // the remaining part of the loop
    const uint _L = f_mgr_dstH.is_left ? f_mgr_dstH.frag_left_over : f_mgr_dstH.frag_len;

    decx::conv_I2R::_im2row_caller_fp32(loc_src,                            row_buf.ptr, 
                                        I2C_buf.ptr,                        kernel->dpitch / 4, 
                                        ker_dims,                           kernel->wpitch, 
                                        src_buf_dp_x_wp,                    I2C_buf_dim.x, 
                                        make_uint2(src->wpitch, _L),        &t1D);

    decx::conv_I2R::_im2row_eq_GEMM_caller_fp32(I2C_buf.ptr,                (float*)kernel->TensArr.ptr, 
                                                loc_dst,                    I2C_buf_dim.x, 
                                                src->wpitch * _L,           dst->dpitch, 
                                                kernel->tensor_num,         &t1D);

    // deallocate the spaces
    decx::alloc::_host_virtual_page_dealloc(&I2C_buf);
    decx::alloc::_host_virtual_page_dealloc(&src_buf);
    decx::alloc::_host_virtual_page_dealloc(&row_buf);
}
