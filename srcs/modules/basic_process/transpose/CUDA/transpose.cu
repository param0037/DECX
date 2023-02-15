/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "transpose_kernel.cuh"
#include "../../../classes/GPU_Matrix.h"


namespace decx
{
    namespace bp {
        static void dev_transpose_4x4(const float4* src, float4* dst, const uint2 proc_dims, const uint Wsrc, const uint Wdst, decx::cuda_stream* S);


        static void dev_transpose_2x2(const double2* src, double2* dst, const uint2 proc_dims, const uint Wsrc, const uint Wdst, decx::cuda_stream* S);


        static void dev_transpose_8x8(const float4* src, float4* dst, const uint2 proc_dims, const uint Wsrc, const uint Wdst, decx::cuda_stream* S);
    }
}


static void 
decx::bp::dev_transpose_4x4(const float4* src, float4* dst, const uint2 proc_dims, const uint Wsrc, const uint Wdst, decx::cuda_stream* S)
{
    const dim3 grid(decx::utils::ceil<int>(proc_dims.y, 16 * 4),
        decx::utils::ceil<int>(proc_dims.x, 16 * 4));
    const dim3 block(16, 16);

    decx::bp::GPUK::cu_transpose_vec4x4 << <grid, block, 0, S->get_raw_stream_ref() >> > (
        src, dst, Wsrc / 4, Wdst / 4, proc_dims);
}




static void
decx::bp::dev_transpose_2x2(const double2* src, double2* dst, const uint2 proc_dims, const uint Wsrc, const uint Wdst, decx::cuda_stream* S)
{
    const dim3 grid(decx::utils::ceil<int>(proc_dims.y, 16 * 2),
        decx::utils::ceil<int>(proc_dims.x, 16 * 2));
    const dim3 block(16, 16);

    decx::bp::GPUK::cu_transpose_vec2x2 << <grid, block, 0, S->get_raw_stream_ref() >> > (
        src, dst, Wsrc / 2, Wdst / 2, proc_dims);
}




static void
decx::bp::dev_transpose_8x8(const float4* src, float4* dst, const uint2 proc_dims, const uint Wsrc, const uint Wdst, decx::cuda_stream* S)
{
    const dim3 grid(decx::utils::ceil<int>(proc_dims.y, 16 * 8),
        decx::utils::ceil<int>(proc_dims.x, 16 * 8));
    const dim3 block(16, 16);

    decx::bp::GPUK::cu_transpose_vec8x8 << <grid, block, 0, S->get_raw_stream_ref() >> > (
        src, dst, Wsrc / 8, Wdst / 8, proc_dims);
}




namespace de
{
    namespace cuda {
        _DECX_API_ de::DH Transpose(de::GPU_Matrix& src, de::GPU_Matrix& dst);
    }
}


_DECX_API_ de::DH
de::cuda::Transpose(de::GPU_Matrix& src, de::GPU_Matrix& dst)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        Print_Error_Message(4, CUDA_NOT_INIT);
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    decx::Success(&handle);

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);

    if (_src->type == decx::_DATA_TYPES_FLAGS_::_FP32_ || _src->type == decx::_DATA_TYPES_FLAGS_::_INT32_) {
        decx::bp::dev_transpose_4x4((float4*)_src->Mat.ptr, (float4*)_dst->Mat.ptr,
            make_uint2(_src->width, _src->height), _src->pitch, _dst->pitch, S);
    }
    else if (_src->type == decx::_DATA_TYPES_FLAGS_::_FP64_ || _src->type == decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        decx::bp::dev_transpose_2x2((double2*)_src->Mat.ptr, (double2*)_dst->Mat.ptr,
            make_uint2(_src->width, _src->height), _src->pitch, _dst->pitch, S);
    }
    else if (_src->type == decx::_DATA_TYPES_FLAGS_::_FP16_) {
        decx::bp::dev_transpose_8x8((float4*)_src->Mat.ptr, (float4*)_dst->Mat.ptr,
            make_uint2(_src->width, _src->height), _src->pitch, _dst->pitch, S);
    }
    else {
        Print_Error_Message(4, MEANINGLESS_FLAG);
        decx::err::InvalidParam(&handle);
        S->detach();
        return handle;
    }

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();

    return handle;
}
