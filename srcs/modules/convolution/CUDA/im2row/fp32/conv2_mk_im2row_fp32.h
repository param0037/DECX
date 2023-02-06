/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_MK_IM2ROW_FP32_H_
#define _CONV2_MK_IM2ROW_FP32_H_

#include "../../../../classes/Tensor.h"
#include "../../../../classes/TensorArray.h"
#include "../im2row.cuh"
#include "eq_GEMM_fp32.cuh"
#include "../../../conv_utils.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../GPU_conv2_utils.cuh"



namespace decx
{
    namespace conv_I2R {
        static void sconv2_mk_im2row_frag(float4* src_buf, float4* ker_buf, float4* I2C_buf, float4* dst,
            decx::conv_I2R::_conv2_I2C_params_set* _im2col_params, decx::cuda_stream* S);


        static void sconv2_NB_r8_mk_im2row(
            decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel, decx::_GPU_Tensor* dst, decx::cuda_stream* S);



        static void sconv2_BC_r8_mk_im2row(
            decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel, decx::_GPU_Tensor* dst, decx::cuda_stream* S);
    }
}




static void decx::conv_I2R::sconv2_mk_im2row_frag(float4* src_buf, float4* ker_buf, float4* I2C_buf, float4* dst,
    decx::conv_I2R::_conv2_I2C_params_set* _im2col_params, decx::cuda_stream* S)
{
    const uint proc_H = _im2col_params->src_proc_H;

    const size_t I2C_dimH = _im2col_params->Wdst_o * proc_H;

    int2 thread_bounds = make_int2(_im2col_params->Wdst_o / 4, proc_H);
    dim3 block_I2C(16, 16);
    dim3 grid_I2C(decx::utils::ceil<int>(thread_bounds.y, 16), decx::utils::ceil<int>(thread_bounds.x, 16));
    const int2 kernel_shift = make_int2(8 - _im2col_params->ker_dims.y / 2, 8 - _im2col_params->ker_dims.x / 2);

    decx::conv_I2R::GPUK::cu_Im2Row_v128_r8_within << <grid_I2C, block_I2C, 0, S->get_raw_stream_ref() >> > (src_buf,
                                                                 I2C_buf,
                                                                 kernel_shift,
                                                                 thread_bounds,
                                                                 _im2col_params->Wsrc_buf / 4,
                                                                 _im2col_params->WI2C_buf / 4,
                                                                 _im2col_params->ker_dims,
                                                                 _im2col_params->depth / 4);

    const ulong2 MatA_load_bounds = make_ulong2(_im2col_params->WI2C_buf / 4, I2C_dimH / 4);

    dim3 block_eqMM(32, 8);
    dim3 grid_eqMM(decx::utils::ceil<size_t>(I2C_dimH, 128),
                   decx::utils::ceil<uint>(_im2col_params->k_tensor_num, 32));

    decx::conv_I2R::GPUK::cu_conv_eq_mm_fp32 << <grid_eqMM, block_eqMM, 0, S->get_raw_stream_ref() >> > (I2C_buf,
                                                            ker_buf,
                                                            dst,
                                                            MatA_load_bounds,
                                                            _im2col_params->Wdst_eqMM / 4,
                                                            _im2col_params->ker_buf_dim.x / 4,
                                                            _im2col_params->ker_buf_dim.x / 128);
}


#define _I2C_size_ 1024 * 1024 * 4



static void decx::conv_I2R::sconv2_NB_r8_mk_im2row(
    decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel, decx::_GPU_Tensor* dst, decx::cuda_stream *S)
{
    const int2 kernel_shift = make_int2(8 - kernel->height / 2, 8 - kernel->width / 2);
    
    // the width and height of output tensor
    const uint4 dst_o_dim = make_uint4(src->width - (kernel->width / 2) * 2, 
                                       src->height - (kernel->height / 2) * 2, 
                                       kernel->tensor_num,
                                       decx::DATA_STORE_TYPE::Page_Locked);
    
    dst->re_construct(src->type,
                      src->width - (kernel->width / 2) * 2,
                      src->height - (kernel->height / 2) * 2,
                      kernel->tensor_num);

    // the dimensions of kernel buffer, width : the number of active values in kernel, but dpitch included
    // height : the number of tensors 
    const int2 ker_buf_dim = make_int2(decx::utils::ceil<uint>(kernel->plane[0] * (size_t)kernel->dpitch, 128) * 128, 
                                       decx::utils::ceil<uint>(kernel->tensor_num, 32) * 32);

    // the dimension of the matrix after im2col operation
    const int2 eq_src_dims = make_int2(decx::utils::ceil<uint>(dst_o_dim.x, 4) * 4, 
                                       dst_o_dim.y);

    // the dimensions of src buffer
    const ulong2 src_buf_dim = make_ulong2((decx::utils::ceil<size_t>(src->width, 64) * 64 + kernel_shift.y * 2) * (size_t)src->dpitch,
        decx::utils::ceil<uint>(src->height, 16) * 16 + kernel_shift.x * 2);

    const ulong2 I2C_dims = make_ulong2(kernel->plane[0] * (size_t)kernel->dpitch, 
                                        eq_src_dims.x * eq_src_dims.y);

    const uint frag_num = ((size_t)dst_o_dim.y / decx::utils::ceil<size_t>(I2C_dims.x * I2C_dims.y, _I2C_size_));
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, dst_o_dim.y, frag_num);

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(kernel->tensor_num, 32) * 32, 
                                         decx::utils::ceil<size_t>(I2C_dims.y, 128) * 128);
    
    const uint I2C_alloc_height = f_mgr.frag_len + 16;

    decx::PtrInfo<float4> src_buf, I2C_buf, ker_buf;
    if (decx::alloc::_device_malloc(&src_buf, src_buf_dim.x * src_buf_dim.y * sizeof(float), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&I2C_buf, I2C_dims.x * I2C_alloc_height * sizeof(float), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&ker_buf, ker_buf_dim.x * ker_buf_dim.y * sizeof(float), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    
    // copy data from kernel(host) to kernel_buffer(device)
    for (int i = 0; i < kernel->tensor_num; ++i) {
        cudaMemcpy2DAsync(reinterpret_cast<float*>(ker_buf.ptr) + i * ker_buf_dim.x,                             kernel->dpitch * kernel->width * sizeof(float),
                          kernel->TensptrArr.ptr[i],                                                             kernel->dp_x_wp * sizeof(float),
                          kernel->dpitch * kernel->width * sizeof(float),                                        kernel->height,
                          cudaMemcpyDeviceToDevice,                                                              S->get_raw_stream_ref());
    }

    // copy data from src(host) to src_buffer(device)
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(src_buf.ptr) + kernel_shift.x * src_buf_dim.x + kernel_shift.y * src->dpitch,     src_buf_dim.x * sizeof(float), 
        src->Tens.ptr,                                                                                             src->dp_x_wp * sizeof(float),
        src->dp_x_wp * sizeof(float),                                                                              src->height, 
        cudaMemcpyDeviceToDevice,                                                                                  S->get_raw_stream_ref()));
    

    decx::conv_I2R::_conv2_I2C_params_set _params;
    _params.depth           = kernel->dpitch;
    _params.ker_buf_dim     = ker_buf_dim;
    _params.ker_dims        = make_int2(kernel->width, kernel->height);
    _params.k_tensor_num    = kernel->tensor_num;
    _params.src_proc_H      = f_mgr.frag_len;
    _params.Wdst_eqMM       = decx::utils::ceil<uint>(kernel->tensor_num, 4) * 4;
    _params.WI2C_buf        = I2C_dims.x;
    _params.Wsrc_buf        = src_buf_dim.x;
    _params.Wdst_o          = decx::utils::ceil<uint>(dst_o_dim.x, 4) * 4;

    float4* loc_src_buf = src_buf.ptr, * loc_dst = (float4*)dst->Tens.ptr;
    for (int i = 0; i < f_mgr.frag_num - 1; ++i) 
    {
        decx::conv_I2R::sconv2_mk_im2row_frag(loc_src_buf, ker_buf.ptr, I2C_buf.ptr, loc_dst, &_params, S);
        loc_src_buf += f_mgr.frag_len * _params.Wsrc_buf / 4;
        loc_dst += f_mgr.frag_len * _params.Wdst_o * dst->dpitch / 4;
    }
    _params.src_proc_H = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    decx::conv_I2R::sconv2_mk_im2row_frag(loc_src_buf, ker_buf.ptr, I2C_buf.ptr, loc_dst, &_params, S);

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&src_buf);
    decx::alloc::_device_dealloc(&ker_buf);
    decx::alloc::_device_dealloc(&I2C_buf);
}




static void decx::conv_I2R::sconv2_BC_r8_mk_im2row(
    decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel, decx::_GPU_Tensor* dst, decx::cuda_stream *S)
{
    const int2 kernel_shift = make_int2(8 - kernel->height / 2, 8 - kernel->width / 2);
    
    // the width and height of output tensor
    const uint4 dst_o_dim = make_uint4(src->width, src->height, kernel->tensor_num, 0);
    // re-construct destinated matrix to match the shapes
    dst->re_construct(src->type, src->width, src->height, kernel->tensor_num);

    // the dimensions of kernel buffer, width : the number of active values in kernel, but dpitch included
    // height : the number of tensors 
    const int2 ker_buf_dim = make_int2(decx::utils::ceil<uint>(kernel->plane[0] * (size_t)kernel->dpitch, 128) * 128, 
                                         decx::utils::ceil<uint>(kernel->tensor_num, 32) * 32);

    // the theoretical dimension of the source matrix (acts as boundaries in im2col operation)
    const int2 eq_src_dims = make_int2(decx::utils::ceil<uint>(dst_o_dim.x, 4) * 4, 
                                       dst_o_dim.y);

    // the dimensions of src buffer
    const ulong2 src_buf_dim = make_ulong2((decx::utils::ceil<size_t>(src->width, 64) * 64 + 16) * (size_t)src->dpitch, 
                                            decx::utils::ceil<uint>(src->height, 16) * 16 + 16);
    
    const ulong2 I2C_dims = make_ulong2(kernel->plane[0] * (size_t)kernel->dpitch, 
                                        eq_src_dims.x * eq_src_dims.y);

    const uint frag_num = ((size_t)dst_o_dim.y / decx::utils::ceil<size_t>(I2C_dims.x * I2C_dims.y, _I2C_size_));
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, dst_o_dim.y, frag_num);

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(kernel->tensor_num, 32) * 32, 
                                         decx::utils::ceil<size_t>(I2C_dims.y, 128) * 128);
    const uint I2C_alloc_height = f_mgr.frag_len + 16;
    
    decx::PtrInfo<float4> src_buf, I2C_buf, ker_buf;
    if (decx::alloc::_device_malloc(&src_buf, src_buf_dim.x * src_buf_dim.y * sizeof(float), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&I2C_buf, I2C_dims.x * I2C_alloc_height * sizeof(float), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&ker_buf, ker_buf_dim.x * ker_buf_dim.y * sizeof(float), true, S)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    // copy data from kernel(host) to kernel_buffer(device)
    for (int i = 0; i < kernel->tensor_num; ++i) {
        cudaMemcpy2DAsync(reinterpret_cast<float*>(ker_buf.ptr) + i * ker_buf_dim.x,                kernel->dpitch * kernel->width * sizeof(float),
                          kernel->TensptrArr.ptr[i],                                                kernel->dp_x_wp * sizeof(float),
                          kernel->dpitch * kernel->width * sizeof(float),                           kernel->height,
                          cudaMemcpyDeviceToDevice,                                                 S->get_raw_stream_ref());
    }

    // copy data from src(host) to src_buffer(device)
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(src_buf.ptr) + 8 * src_buf_dim.x + 8 * src->dpitch,                src_buf_dim.x * sizeof(float), 
        src->Tens.ptr,                                                                              src->dp_x_wp * sizeof(float),
        src->dp_x_wp * sizeof(float),                                                               src->height, 
        cudaMemcpyDeviceToDevice,                                                                   S->get_raw_stream_ref()));


    decx::conv_I2R::_conv2_I2C_params_set _params;
    _params.depth           = kernel->dpitch;
    _params.ker_buf_dim     = ker_buf_dim;
    _params.ker_dims        = make_int2(kernel->width, kernel->height);
    _params.k_tensor_num    = kernel->tensor_num;
    _params.src_proc_H      = f_mgr.frag_len;
    _params.Wdst_eqMM       = decx::utils::ceil<uint>(kernel->tensor_num, 4) * 4;
    _params.WI2C_buf        = I2C_dims.x;
    _params.Wsrc_buf        = src_buf_dim.x;
    _params.Wdst_o          = decx::utils::ceil<uint>(dst_o_dim.x, 4) * 4;

    float4* loc_src_buf = src_buf.ptr, *loc_dst = (float4*)dst->Tens.ptr;
    for (int i = 0; i < f_mgr.frag_num - 1; ++i) 
    {
        decx::conv_I2R::sconv2_mk_im2row_frag(loc_src_buf, ker_buf.ptr, I2C_buf.ptr, loc_dst, &_params, S);
        loc_src_buf += f_mgr.frag_len * _params.Wsrc_buf / 4;
        loc_dst += f_mgr.frag_len * _params.Wdst_o * dst->dpitch / 4;
    }
    _params.src_proc_H = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    decx::conv_I2R::sconv2_mk_im2row_frag(loc_src_buf, ker_buf.ptr, I2C_buf.ptr, loc_dst, &_params, S);

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&src_buf);
    decx::alloc::_device_dealloc(&ker_buf);
    decx::alloc::_device_dealloc(&I2C_buf);
}


#endif
