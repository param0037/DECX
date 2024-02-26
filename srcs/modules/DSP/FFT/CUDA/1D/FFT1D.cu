/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "CUDA_FFT1D_planner.cuh"
#include "../CUDA_FFTs.cuh"


namespace decx
{
namespace dsp {
    namespace fft 
    {
        template <typename _type_in> _CRSR_ 
        static void _FFT1D_cplxf32_on_GPU(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle);


        template <typename _type_out> _CRSR_ 
        static void _IFFT1D_cplxf32_on_GPU(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle);


        template <typename _type_in> _CRSR_
            static void _FFT1D_cplxf32(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);


        template <typename _type_out> _CRSR_
            static void _IFFT1D_cplxf32(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);
    }
}
}


template <typename _type_in> _CRSR_ 
static void decx::dsp::fft::_FFT1D_cplxf32_on_GPU(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    //decx::dsp::fft::_cuda_FFT1D_planner<float> _planner;
    if (decx::dsp::fft::cuda_FFT1D_cplxf32_planner == NULL) {
        decx::dsp::fft::cuda_FFT1D_cplxf32_planner = new decx::dsp::fft::_cuda_FFT1D_planner<float>;
    }
    if (decx::dsp::fft::cuda_FFT1D_cplxf32_planner->changed(src->Len())) {
        decx::dsp::fft::cuda_FFT1D_cplxf32_planner->plan(src->Len(), handle, S);
        Check_Runtime_Error(handle);
    }

    /*const decx::dsp::fft::_cuda_FFT2D_planner<float>* _formal_FFT2D_ptr = _planner.get_FFT2D_planner();
    decx::utils::double_buffer_manager _double_buffer(_formal_FFT2D_ptr->get_tmp1_ptr<void>(),
                                                      _formal_FFT2D_ptr->get_tmp2_ptr<void>());

    decx::dsp::fft::FFT1D_partition_cplxf_1st_caller<_type_in, false>(src->Vec.ptr, &_double_buffer,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S);

    decx::bp::transpose2D_b8_for_FFT(_double_buffer.get_leading_ptr<double2>(),
                                     _double_buffer.get_lagging_ptr<double2>(),
                                     make_uint2(_planner.get_larger_FFT_lengths(0), _planner.get_larger_FFT_lengths(1)),
                                     _formal_FFT2D_ptr->get_buffer_dims().x,
                                     _formal_FFT2D_ptr->get_buffer_dims().y, S);

    _double_buffer.update_states();

    decx::dsp::fft::FFT1D_partition_cplxf_end_caller<_FFT1D_END_>(&_double_buffer, dst->Vec.ptr,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);*/
    decx::dsp::fft::cuda_FFT1D_cplxf32_planner->Forward<_type_in>(src, dst, S);

    E->event_record(S);
    S->synchronize();

    S->detach();
    E->detach();
}



template <typename _type_in> _CRSR_ 
static void decx::dsp::fft::_FFT1D_cplxf32(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    //decx::dsp::fft::_cuda_FFT1D_planner<float> _planner;
    if (decx::dsp::fft::cuda_FFT1D_cplxf32_planner == NULL) {
        decx::dsp::fft::cuda_FFT1D_cplxf32_planner = new decx::dsp::fft::_cuda_FFT1D_planner<float>;
    }
    if (decx::dsp::fft::cuda_FFT1D_cplxf32_planner->changed(src->Len())) {
        decx::dsp::fft::cuda_FFT1D_cplxf32_planner->plan(src->Len(), handle, S);
        Check_Runtime_Error(handle);
    }

    /*const decx::dsp::fft::_cuda_FFT2D_planner<float>* _formal_FFT2D_ptr = _planner.get_FFT2D_planner();
    decx::utils::double_buffer_manager _double_buffer(_formal_FFT2D_ptr->get_tmp1_ptr<void>(),
                                                      _formal_FFT2D_ptr->get_tmp2_ptr<void>());

    checkCudaErrors(cudaMemcpyAsync(_double_buffer.get_buffer2<void>(), src->Vec.ptr, src->Len() * sizeof(_type_in),
        cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::dsp::fft::FFT1D_partition_cplxf_1st_caller<_type_in, false>(NULL, &_double_buffer,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S);

    decx::bp::transpose2D_b8_for_FFT(_double_buffer.get_leading_ptr<double2>(),
                                     _double_buffer.get_lagging_ptr<double2>(),
                                     make_uint2(_planner.get_larger_FFT_lengths(0), _planner.get_larger_FFT_lengths(1)),
                                     _formal_FFT2D_ptr->get_buffer_dims().x,
                                     _formal_FFT2D_ptr->get_buffer_dims().y, S);

    _double_buffer.update_states();

    decx::dsp::fft::FFT1D_partition_cplxf_end_caller<_FFT1D_END_>(&_double_buffer, NULL,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _double_buffer.get_leading_ptr<void>(), src->Len() * sizeof(de::CPf),
        cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));*/

    decx::dsp::fft::cuda_FFT1D_cplxf32_planner->Forward<_type_in>(src, dst, S);

    E->event_record(S);
    S->synchronize();

    S->detach();
    E->detach();
}



template <typename _type_out> _CRSR_ 
static void decx::dsp::fft::_IFFT1D_cplxf32_on_GPU(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    //decx::dsp::fft::_cuda_FFT1D_planner<float> _planner;
    if (decx::dsp::fft::cuda_IFFT1D_cplxf32_planner == NULL) {
        decx::dsp::fft::cuda_IFFT1D_cplxf32_planner = new decx::dsp::fft::_cuda_FFT1D_planner<float>;
    }
    if (decx::dsp::fft::cuda_IFFT1D_cplxf32_planner->changed(src->Len())) {
        decx::dsp::fft::cuda_IFFT1D_cplxf32_planner->plan(src->Len(), handle, S);
        Check_Runtime_Error(handle);
    }

    /*const decx::dsp::fft::_cuda_FFT2D_planner<float>* _formal_FFT2D_ptr = _planner.get_FFT2D_planner();
    decx::utils::double_buffer_manager _double_buffer(_formal_FFT2D_ptr->get_tmp1_ptr<void>(),
                                                      _formal_FFT2D_ptr->get_tmp2_ptr<void>());

    decx::dsp::fft::FFT1D_partition_cplxf_1st_caller<de::CPf, true>(src->Vec.ptr, &_double_buffer,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S, _planner.get_signal_length());

    decx::bp::transpose2D_b8_for_FFT(_double_buffer.get_leading_ptr<double2>(),
                                     _double_buffer.get_lagging_ptr<double2>(),
                                     make_uint2(_planner.get_larger_FFT_lengths(0), _planner.get_larger_FFT_lengths(1)),
                                     _formal_FFT2D_ptr->get_buffer_dims().x,
                                     _formal_FFT2D_ptr->get_buffer_dims().y, S);

    _double_buffer.update_states();

    decx::dsp::fft::FFT1D_partition_cplxf_end_caller<_IFFT1D_END_(_type_out)>(&_double_buffer, dst->Vec.ptr,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);*/

    decx::dsp::fft::cuda_IFFT1D_cplxf32_planner->Inverse<_type_out>(src, dst, S);

    E->event_record(S);
    S->synchronize();

    S->detach();
    E->detach();
}



template <typename _type_out> _CRSR_ 
static void decx::dsp::fft::_IFFT1D_cplxf32(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    //decx::dsp::fft::_cuda_FFT1D_planner<float> _planner;
    if (decx::dsp::fft::cuda_IFFT1D_cplxf32_planner == NULL) {
        decx::dsp::fft::cuda_IFFT1D_cplxf32_planner = new decx::dsp::fft::_cuda_FFT1D_planner<float>;
    }
    if (decx::dsp::fft::cuda_IFFT1D_cplxf32_planner->changed(src->Len())) {
        decx::dsp::fft::cuda_IFFT1D_cplxf32_planner->plan(src->Len(), handle, S);
        Check_Runtime_Error(handle);
    }

    /*const decx::dsp::fft::_cuda_FFT2D_planner<float>* _formal_FFT2D_ptr = _planner.get_FFT2D_planner();
    decx::utils::double_buffer_manager _double_buffer(_formal_FFT2D_ptr->get_tmp1_ptr<void>(),
                                                      _formal_FFT2D_ptr->get_tmp2_ptr<void>());

    checkCudaErrors(cudaMemcpyAsync(_double_buffer.get_buffer2<void>(), src->Vec.ptr, src->Len() * sizeof(de::CPf),
        cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::dsp::fft::FFT1D_partition_cplxf_1st_caller<de::CPf, true>(NULL, &_double_buffer,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S, _planner.get_signal_length());

    decx::bp::transpose2D_b8_for_FFT(_double_buffer.get_leading_ptr<double2>(),
                                     _double_buffer.get_lagging_ptr<double2>(),
                                     make_uint2(_planner.get_larger_FFT_lengths(0), _planner.get_larger_FFT_lengths(1)),
                                     _formal_FFT2D_ptr->get_buffer_dims().x,
                                     _formal_FFT2D_ptr->get_buffer_dims().y, S);

    _double_buffer.update_states();

    decx::dsp::fft::FFT1D_partition_cplxf_end_caller<_IFFT1D_END_(_type_out)>(&_double_buffer, NULL,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _double_buffer.get_leading_ptr<void>(), src->Len() * sizeof(_type_out),
        cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));*/

    decx::dsp::fft::cuda_IFFT1D_cplxf32_planner->Inverse<_type_out>(src, dst, S);

    E->event_record(S);
    S->synchronize();

    S->detach();
    E->detach();
}



_DECX_API_ de::DH de::dsp::cuda::FFT(de::GPU_Vector& src, de::GPU_Vector& dst)
{
    de::DH handle;

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_FFT1D_cplxf32_on_GPU<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_FFT1D_cplxf32_on_GPU<de::CPf>(_src, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}



_DECX_API_ de::DH de::dsp::cuda::FFT(de::Vector& src, de::Vector& dst)
{
    de::DH handle;

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_FFT1D_cplxf32<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_FFT1D_cplxf32<de::CPf>(_src, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}




_DECX_API_ de::DH de::dsp::cuda::IFFT(de::GPU_Vector& src, de::GPU_Vector& dst, const de::_DATA_TYPES_FLAGS_ _type_out)
{
    de::DH handle;

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    switch (_type_out)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_IFFT1D_cplxf32_on_GPU<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_IFFT1D_cplxf32_on_GPU<de::CPf>(_src, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}




_DECX_API_ de::DH de::dsp::cuda::IFFT(de::Vector& src, de::Vector& dst, const de::_DATA_TYPES_FLAGS_ _type_out)
{
    de::DH handle;

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (_type_out)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_IFFT1D_cplxf32<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_IFFT1D_cplxf32<de::CPf>(_src, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}



void decx::dsp::InitCUDA_FFT1D_Resources()
{
    decx::dsp::fft::cuda_FFT1D_cplxf32_planner = NULL;
    decx::dsp::fft::cuda_IFFT1D_cplxf32_planner = NULL;
}



void decx::dsp::FreeCUDA_FFT1D_Resources()
{
    if (decx::dsp::fft::cuda_FFT1D_cplxf32_planner != NULL) {
        delete decx::dsp::fft::cuda_FFT1D_cplxf32_planner;
    }
    if (decx::dsp::fft::cuda_IFFT1D_cplxf32_planner != NULL) {
        delete decx::dsp::fft::cuda_IFFT1D_cplxf32_planner;
    }
}
