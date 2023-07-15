/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "conv2_mk_im2col_fp16.h"


template <bool _print>
void decx::conv_I2R::conv2_BC_im2col_fp16(decx::_GPU_Tensor*         src,
                                      decx::_GPU_TensorArray*    kernel, 
                                      decx::_GPU_Tensor*         dst, 
                                      const int                  accu_flag, 
                                      decx::cuda_stream*         S,
                                      decx::cuda_event*          E,
                                      de::DH*                    handle)
{
    const int2 kernel_shift = make_int2(8 - kernel->get_layout().height / 2, 8 - kernel->get_layout().width / 2);
    
    // the width and height of output tensor
    const uint4 dst_o_dim = make_uint4(src->get_layout().width, 
                                       src->get_layout().height,
                                       kernel->TensorNum(), 0);

    dst->re_construct(src->Type(), src->get_layout().width, src->get_layout().height, kernel->TensorNum());

    // the dimensions of kernel buffer, width : the number of active values in kernel, but dpitch included
    // height : the number of tensors 
    const int2 ker_buf_dim = make_int2(decx::utils::ceil<uint>(kernel->get_layout().plane[0] * (size_t)kernel->get_layout().dpitch, 128) * 128,
                                         decx::utils::ceil<uint>(kernel->TensorNum(), 64) * 64);

    // the dimension of the matrix after im2col operation
    const int2 eq_src_dims = make_int2(decx::utils::ceil<uint>(dst_o_dim.x, 8) * 8, dst_o_dim.y);

    // the dimensions of src buffer
    const ulong2 src_buf_dim = make_ulong2((decx::utils::ceil<size_t>(src->get_layout().width + kernel->get_layout().width - 1, 8) * 8) * (size_t)src->get_layout().dpitch,
                                           src->get_layout().height + kernel->get_layout().height - 1);
    
    const ulong2 I2C_dims = make_ulong2(eq_src_dims.x * eq_src_dims.y,
                                        kernel->get_layout().plane[0] * (size_t)kernel->get_layout().dpitch);

    const uint frag_num = decx::utils::ceil<size_t>(I2C_dims.x * I2C_dims.y, _I2C_size_fp16_);
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, dst_o_dim.y, frag_num);

    const uint I2C_alloc_width = decx::utils::ceil<uint32_t>(dst_o_dim.x, 8) * 8 * max(f_mgr.frag_len, f_mgr.frag_left_over);
    
    decx::PtrInfo<float4> src_buf, I2C_buf, ker_buf;
    if (decx::alloc::_device_malloc(&src_buf, src_buf_dim.x * src_buf_dim.y * sizeof(de::Half), true, S)) {
        decx::err::device_AllocateFailure<_print>(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&I2C_buf, I2C_dims.y * I2C_alloc_width * sizeof(de::Half), true, S)) {
        decx::err::device_AllocateFailure<_print>(handle);
        return;
    }
    
    if (decx::alloc::_device_malloc(&ker_buf, ker_buf_dim.x * ker_buf_dim.y * sizeof(de::Half), true, S)) {
        decx::err::device_AllocateFailure<_print>(handle);
        return;
    }
    
    // copy data from kernel(host) to kernel_buffer(device)
    for (int i = 0; i < kernel->TensorNum(); ++i) {
        cudaMemcpy2DAsync(DECX_PTR_SHF_XY<float4, de::Half>(ker_buf.ptr, i, 0, ker_buf_dim.x),      kernel->get_layout().dpitch * kernel->get_layout().width * sizeof(de::Half),
                          kernel->TensptrArr.ptr[i],                                                kernel->get_layout().dp_x_wp * sizeof(de::Half),
                          kernel->get_layout().dpitch * kernel->get_layout().width * sizeof(de::Half),        kernel->get_layout().height,
                          cudaMemcpyHostToDevice,                                                   S->get_raw_stream_ref());
    }

    // copy data from src(host) to src_buffer(device)
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(src_buf.ptr) + (kernel->get_layout().height / 2) * src_buf_dim.x + (kernel->get_layout().width / 2) * src->get_layout().dpitch,
        src_buf_dim.x * sizeof(de::Half), 
        src->Tens.ptr,                                                          src->get_layout().dp_x_wp * sizeof(de::Half),
        src->get_layout().dp_x_wp * sizeof(de::Half),                                src->get_layout().height,
        cudaMemcpyDeviceToDevice,                                               S->get_raw_stream_ref()));

    decx::conv_I2R::_conv2_I2C_params_set _params;
    _params.depth           = kernel->get_layout().dpitch;
    _params.ker_buf_dim     = ker_buf_dim;
    _params.ker_dims        = make_int2(kernel->get_layout().width, kernel->get_layout().height);
    _params.k_tensor_num    = kernel->TensorNum();
    _params.src_proc_H      = f_mgr.frag_len;
    _params.Wdst_eqMM       = decx::utils::ceil<uint>(kernel->TensorNum(), 8) * 8;
    _params.WI2C_buf        = I2C_alloc_width;
    _params.Wsrc_buf        = src_buf_dim.x;
    _params.Wdst_o          = decx::utils::ceil<uint>(dst_o_dim.x, 8) * 8;
    _params.HI2C_buf        = I2C_dims.y;

    float4* loc_src_buf = src_buf.ptr, * loc_dst = (float4*)dst->Tens.ptr;
    for (int i = 0; i < f_mgr.frag_num - 1; ++i)
    {
        decx::conv_I2R::conv2_MK_im2col_frag_fp16(loc_src_buf, ker_buf.ptr, I2C_buf.ptr, loc_dst, &_params, S, accu_flag);
        loc_src_buf += f_mgr.frag_len * _params.Wsrc_buf / 8;
        loc_dst += f_mgr.frag_len * _params.Wdst_o * dst->get_layout().dpitch / 8;
    }
    _params.src_proc_H = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    decx::conv_I2R::conv2_MK_im2col_frag_fp16(loc_src_buf, ker_buf.ptr, I2C_buf.ptr, loc_dst, &_params, S, accu_flag);
    
    E->event_record(S);
    E->synchronize();

    decx::alloc::_device_dealloc(&src_buf);
    decx::alloc::_device_dealloc(&ker_buf);
    decx::alloc::_device_dealloc(&I2C_buf);
}


template void decx::conv_I2R::conv2_BC_im2col_fp16<true>(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel,
    decx::_GPU_Tensor* dst, const int accu_flag, decx::cuda_stream* S, decx::cuda_event* E, de::DH* handle);


template void decx::conv_I2R::conv2_BC_im2col_fp16<true>(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel,
    decx::_GPU_Tensor* dst, const int accu_flag, decx::cuda_stream* S, decx::cuda_event* E, de::DH* handle);




template <bool _print>
void decx::conv_I2R::conv2_BC_im2col_fp16_stride(decx::_GPU_Tensor*         src,
                                             decx::_GPU_TensorArray*    kernel, 
                                             decx::_GPU_Tensor*         dst, 
                                             const int                  accu_flag, 
                                             decx::cuda_stream*         S,
                                             decx::cuda_event*          E,
                                             const uint2                strideXY,
                                             de::DH*                    handle)
{
    // the width and height of output tensor
    const uint4 dst_o_dim = make_uint4(src->get_layout().width / strideXY.x,
                                       src->get_layout().height / strideXY.y,
                                       kernel->TensorNum(), 0);
    
    dst->re_construct(src->Type(), dst_o_dim.x, dst_o_dim.y, kernel->TensorNum());

    // the dimensions of kernel buffer, width : the number of active values in kernel, but dpitch included
    // height : the number of tensors 
    const int2 ker_buf_dim = make_int2(decx::utils::ceil<uint>(kernel->get_layout().plane[0] * (size_t)kernel->get_layout().dpitch, 128) * 128,
                                         decx::utils::ceil<uint>(kernel->TensorNum(), 64) * 64);

    // the dimension of the matrix after im2col operation
    const int2 eq_src_dims = make_int2(decx::utils::ceil<uint>(dst_o_dim.x, 8) * 8, dst_o_dim.y);

    // the dimensions of src buffer
    const ulong2 src_buf_dim = make_ulong2((decx::utils::ceil<size_t>(src->get_layout().width + kernel->get_layout().width - 1, 8) * 8) * (size_t)src->get_layout().dpitch,
                                            src->get_layout().height + dst->get_layout().height - 1);
    
    const ulong2 I2C_dims = make_ulong2(eq_src_dims.x * eq_src_dims.y,
                                        kernel->get_layout().plane[0] * (size_t)kernel->get_layout().dpitch);

    const uint frag_num = decx::utils::ceil<size_t>(I2C_dims.x * I2C_dims.y, _I2C_size_fp16_);
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, dst_o_dim.y, frag_num);

    const uint I2C_alloc_width = decx::utils::ceil<uint32_t>(dst_o_dim.x, 8) * 8 * max(f_mgr.frag_len, f_mgr.frag_left_over);
    
    decx::PtrInfo<float4> src_buf, I2C_buf, ker_buf;
    if (decx::alloc::_device_malloc(&src_buf, src_buf_dim.x * src_buf_dim.y * sizeof(de::Half), true, S)) {
        decx::err::device_AllocateFailure<_print>(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&I2C_buf, I2C_dims.y * I2C_alloc_width * sizeof(de::Half), true, S)) {
        decx::err::device_AllocateFailure<_print>(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&ker_buf, ker_buf_dim.x * ker_buf_dim.y * sizeof(de::Half), true, S)) {
        decx::err::device_AllocateFailure<_print>(handle);
        return;
    }
    
    // copy data from kernel(host) to kernel_buffer(device)
    for (int i = 0; i < kernel->TensorNum(); ++i) {
        cudaMemcpy2DAsync(DECX_PTR_SHF_XY<float4, de::Half>(ker_buf.ptr, i, 0, ker_buf_dim.x),      kernel->get_layout().dpitch * kernel->get_layout().width * sizeof(de::Half),
                          kernel->TensptrArr.ptr[i],                                                kernel->get_layout().dp_x_wp * sizeof(de::Half),
                          kernel->get_layout().dpitch * kernel->get_layout().width * sizeof(de::Half),        kernel->get_layout().height,
                          cudaMemcpyHostToDevice,                                                   S->get_raw_stream_ref());
    }

    // copy data from src(host) to src_buffer(device)
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(src_buf.ptr) + (kernel->get_layout().height / 2) * src_buf_dim.x + (kernel->get_layout().width / 2) * src->get_layout().dpitch,
        src_buf_dim.x * sizeof(de::Half),
        src->Tens.ptr,                                                                  src->get_layout().dp_x_wp * sizeof(de::Half),
        src->get_layout().dp_x_wp * sizeof(de::Half),                                        src->get_layout().height,
        cudaMemcpyDeviceToDevice,                                                       S->get_raw_stream_ref()));

    decx::conv_I2R::_conv2_I2C_params_set _params;
    _params.depth           = kernel->get_layout().dpitch;
    _params.ker_buf_dim     = ker_buf_dim;
    _params.ker_dims        = make_int2(kernel->get_layout().width, kernel->get_layout().height);
    _params.k_tensor_num    = kernel->TensorNum();
    _params.src_proc_H      = f_mgr.frag_len;
    _params.Wdst_eqMM       = decx::utils::ceil<uint>(kernel->TensorNum(), 8) * 8;
    _params.WI2C_buf        = I2C_alloc_width;
    _params.Wsrc_buf        = src_buf_dim.x;
    _params.Wdst_o          = decx::utils::ceil<uint>(dst_o_dim.x, 8) * 8;
    _params.HI2C_buf        = I2C_dims.y;
    _params.strideXY        = strideXY;

    float4* loc_src_buf = src_buf.ptr, * loc_dst = (float4*)dst->Tens.ptr;
    for (int i = 0; i < f_mgr.frag_num - 1; ++i)
    {
        decx::conv_I2R::conv2_MK_im2col_frag_fp16_stride(loc_src_buf, ker_buf.ptr, I2C_buf.ptr, loc_dst, &_params, S, accu_flag);
        loc_src_buf += f_mgr.frag_len * _params.Wsrc_buf / 8 * strideXY.y;
        loc_dst     += f_mgr.frag_len * _params.Wdst_o * dst->get_layout().dpitch / 8;
    }
    _params.src_proc_H = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    decx::conv_I2R::conv2_MK_im2col_frag_fp16_stride(loc_src_buf, ker_buf.ptr, I2C_buf.ptr, loc_dst, &_params, S, accu_flag);
    
    E->event_record(S);
    E->synchronize();

    decx::alloc::_device_dealloc(&src_buf);
    decx::alloc::_device_dealloc(&ker_buf);
    decx::alloc::_device_dealloc(&I2C_buf);
}


template void decx::conv_I2R::conv2_BC_im2col_fp16_stride<true>(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel,
    decx::_GPU_Tensor* dst, const int accu_flag, decx::cuda_stream* S, decx::cuda_event* E, const uint2 strideXY, de::DH* handle);


template void decx::conv_I2R::conv2_BC_im2col_fp16_stride<true>(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel,
    decx::_GPU_Tensor* dst, const int accu_flag, decx::cuda_stream* S, decx::cuda_event* E, const uint2 strideXY, de::DH* handle);