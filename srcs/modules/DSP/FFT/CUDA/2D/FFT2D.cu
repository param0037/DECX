/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "../../../../classes/GPU_Matrix.h"
#include "../../../../core/basic.h"

#include "../../FFT_commons.h"

#include "../../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../../core/cudaStream_management/cudaStream_queue.h"

#include "FFT2D_kernels.cuh"
#include "../../../../core/utils/double_buffer.h"
#include "../../../../BLAS/basic_process/transpose/CUDA/transpose_kernels.cuh"
#include "FFT2D_1way_kernel_callers.cuh"


namespace decx
{
namespace dsp {
namespace fft 
{
    template <typename _type_in> _CRSR_
    static void _FFT2D_caller_cplxf(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle);


    template <typename _type_out> _CRSR_
    static void _IFFT2D_caller_cplxf(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle);
}
}
}


template <typename _type_in> _CRSR_
static void decx::dsp::fft::_FFT2D_caller_cplxf(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle)
{
    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);

    decx::dsp::fft::_cuda_FFT2D_planner<float> _planner(make_uint2(src->Width(), src->Height()), handle);
    _planner.plan(src->Pitch(), dst->Pitch());

    decx::utils::double_buffer_manager double_buffer(_planner.get_tmp1_ptr<void>(), _planner.get_tmp2_ptr<void>());

    decx::dsp::fft::FFT2D_cplxf_1st_1way_caller<_type_in, false>(src->Mat.ptr, &double_buffer,
        _planner.get_FFT_info(decx::dsp::fft::_cuda_FFT2D_planner<float>::_FFT_Vertical),
        S);

    decx::bp::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                             double_buffer.get_lagging_ptr<double2>(),
                             make_uint2(_planner.get_buffer_dims().y, _planner.get_buffer_dims().x),
                             _planner.get_buffer_dims().x, 
                             _planner.get_buffer_dims().y, 
                             S);
    double_buffer.update_states();

    decx::dsp::fft::FFT2D_C2C_cplxf_1way_caller<_FFT2D_END_>(&double_buffer,
        _planner.get_FFT_info(decx::dsp::fft::_cuda_FFT2D_planner<float>::_FFT_Horizontal),
        S);

    decx::bp::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                             (double2*)dst->Mat.ptr,
                             make_uint2(dst->Width(), dst->Height()),
                             _planner.get_buffer_dims().y, 
                             dst->Pitch(), S);
    
    E->event_record(S);
    E->synchronize();

    _planner.release_buffers();
}



template <typename _type_out> _CRSR_
static void decx::dsp::fft::_IFFT2D_caller_cplxf(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle)
{
    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);

    decx::dsp::fft::_cuda_FFT2D_planner<float> _planner(make_uint2(src->Width(), src->Height()), handle);
    _planner.plan(src->Pitch(), dst->Pitch());

    decx::utils::double_buffer_manager double_buffer(_planner.get_tmp1_ptr<void>(), _planner.get_tmp2_ptr<void>());

    decx::dsp::fft::FFT2D_cplxf_1st_1way_caller<de::CPf, true>(src->Mat.ptr, &double_buffer,
        _planner.get_FFT_info(decx::dsp::fft::_cuda_FFT2D_planner<float>::_FFT_Vertical),
        S);

    decx::bp::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                             double_buffer.get_lagging_ptr<double2>(),
                             make_uint2(_planner.get_buffer_dims().y, _planner.get_buffer_dims().x),
                             _planner.get_buffer_dims().x, _planner.get_buffer_dims().y, 
                             S);
    double_buffer.update_states();

    decx::dsp::fft::FFT2D_C2C_cplxf_1way_caller<_IFFT2D_END_(_type_out)>(&double_buffer,
        _planner.get_FFT_info(decx::dsp::fft::_cuda_FFT2D_planner<float>::_FFT_Horizontal),
        S);
    if (std::is_same<_type_out, de::CPf>::value) {
        decx::bp::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                                 (double2*)dst->Mat.ptr,
                                 make_uint2(dst->Width(), dst->Height()),
                                 _planner.get_buffer_dims().y, 
                                 dst->Pitch(), S);
    }
    else if (std::is_same<_type_out, uint8_t>::value) {
        decx::bp::transpose2D_b1(double_buffer.get_leading_ptr<uint32_t>(), 
                                 (uint32_t*)dst->Mat.ptr,
                                 make_uint2(dst->Width(), dst->Height()),
                                 _planner.get_buffer_dims().y * 8,  // Times 8 cuz 8 uchars in one de::CPf
                                 dst->Pitch(), S);
    }
    else if (std::is_same<_type_out, float>::value) {
        decx::bp::transpose2D_b4(double_buffer.get_leading_ptr<float2>(), 
                                 (float2*)dst->Mat.ptr,
                                 make_uint2(dst->Width(), dst->Height()),
                                 _planner.get_buffer_dims().y * 2,  // Times 2 cuz 2 floats in one de::CPf
                                 dst->Pitch(), S);
    }
    
    E->event_record(S);
    E->synchronize();

    _planner.release_buffers();
}


namespace de
{
namespace dsp {
namespace cuda
{
    _DECX_API_ de::DH FFT(de::GPU_Matrix& src, de::GPU_Matrix& dst);


    _DECX_API_ de::DH IFFT(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::_DATA_TYPES_FLAGS_ type_out);
}
}
}



_DECX_API_ de::DH de::dsp::cuda::FFT(de::GPU_Matrix& src, de::GPU_Matrix& dst)
{
    de::DH handle;

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_FFT2D_caller_cplxf<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::fft::_FFT2D_caller_cplxf<uint8_t>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_FFT2D_caller_cplxf<de::CPf>(_src, _dst, &handle);
        break;

    default:
        break;
    }
    

    return handle;
}



_DECX_API_ de::DH de::dsp::cuda::IFFT(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::_DATA_TYPES_FLAGS_ type_out)
{
    de::DH handle;

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    switch (type_out) 
    {
    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_IFFT2D_caller_cplxf<de::CPf>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_IFFT2D_caller_cplxf<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::fft::_IFFT2D_caller_cplxf<uint8_t>(_src, _dst, &handle);
        break;
    }
    
    return handle;
}