/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "FFT2D.h"
#include "FFT2D_kernels.h"
#include "../../../../BLAS/basic_process/transpose/CPU/transpose_exec.h"



namespace decx
{
namespace dsp {
    namespace fft 
    {
        template <typename _type_in>
        _CRSR_ static void FFT2D_caller_cplxf(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle);


        template <typename _type_out>
        _CRSR_ static void IFFT2D_caller_cplxf(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle);
    }
}
}



template <typename _type_in>
static void decx::dsp::fft::FFT2D_caller_cplxf(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

    decx::dsp::fft::cpu_FFT2D_planner<float> planner(make_uint2(src->Width(), src->Height()), handle);
    planner.plan(&t1D, handle);

    decx::bp::_cpu_transpose_config<8> _transpose_config_1st(make_uint2(src->Width(), src->Height()), t1D.total_thread);
    decx::bp::_cpu_transpose_config<8> _transpose_config_2nd(make_uint2(src->Height(), src->Width()), t1D.total_thread);

    // Horizontal FFT
    decx::dsp::fft::_FFT2D_H_entire_rows_cplxf<_type_in, false>((_type_in*)src->Mat.ptr,                             
                                                                (double*)planner.get_tmp1_ptr(), 
                                                                &planner,                                            
                                                                src->Pitch(), 
                                                                decx::utils::ceil<uint32_t>(src->Width(), 4) * 4,   
                                                                &t1D, true);
    // Transpose
    decx::bp::transpose_2x2_caller((double*)planner.get_tmp1_ptr(),                     
                                   (double*)planner.get_tmp2_ptr(),
                                   decx::utils::ceil<uint32_t>(src->Width(), 4) * 4,   
                                   decx::utils::ceil<uint32_t>(src->Height(), 4) * 4, 
                                   &_transpose_config_1st,                              
                                   &t1D);
    // Horizontal FFT
    decx::dsp::fft::_FFT2D_H_entire_rows_cplxf<double, true>((double*)planner.get_tmp2_ptr(),       (double*)planner.get_tmp1_ptr(), 
                                               &planner,                                            decx::utils::ceil<uint32_t>(src->Height(), 4) * 4,  
                                               decx::utils::ceil<uint32_t>(src->Height(), 4) * 4,   &t1D, false);
    // Transpose
    decx::bp::transpose_2x2_caller((double*)planner.get_tmp1_ptr(),                     (double*)dst->Mat.ptr,
                                   decx::utils::ceil<uint32_t>(src->Height(), 4) * 4,   dst->Pitch(), 
                                   &_transpose_config_2nd,                              &t1D);

    planner.release_buffers();
}



template <typename _type_out>
static void decx::dsp::fft::IFFT2D_caller_cplxf(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    decx::dsp::fft::cpu_FFT2D_planner<float> planner(make_uint2(src->Width(), src->Height()), handle);
    
    planner.plan(&t1D, handle);
    decx::bp::_cpu_transpose_config<8> _transpose_config_1st(make_uint2(src->Width(), src->Height()), t1D.total_thread);
    decx::bp::_cpu_transpose_config<sizeof(_type_out)> _transpose_config_2nd(make_uint2(src->Height(), src->Width()), t1D.total_thread);
    // Horizontal FFT
    decx::dsp::fft::_IFFT2D_H_entire_rows_cplxf<double>((double*)src->Mat.ptr,                
                                                        (double*)planner.get_tmp1_ptr(), 
                                                        &planner,                                            
                                                        src->Pitch(), 
                                                        decx::utils::ceil<uint32_t>(src->Width(), 4) * 4,    
                                                        &t1D, true);
                                                        
    // Transpose
    decx::bp::transpose_2x2_caller((double*)planner.get_tmp1_ptr(),                     
                                   (double*)planner.get_tmp2_ptr(),
                                   decx::utils::ceil<uint32_t>(src->Width(), 4) * 4,    
                                   decx::utils::ceil<uint32_t>(src->Height(), 4) * 4, 
                                   &_transpose_config_1st,                              
                                   &t1D);
                                   
    // Horizontal FFT
    const uint8_t _STG_alignment = decx::dsp::fft::cpu_FFT2D_planner<float>::get_alignment_FFT_last_dimension<_type_out>();
    decx::dsp::fft::_IFFT2D_H_entire_rows_cplxf<_type_out>((double*)planner.get_tmp2_ptr(),       
                                                           (_type_out*)planner.get_tmp1_ptr(), 
                                                           &planner,                                            
                                                           decx::utils::ceil<uint32_t>(src->Height(), 4) * 4,  
                                                           decx::utils::ceil<uint32_t>(src->Height(), _STG_alignment) * _STG_alignment,
                                                           &t1D, false);
                                                           
    // Transpose
    if constexpr (std::is_same_v<_type_out, double>){
        decx::bp::transpose_2x2_caller((double*)planner.get_tmp1_ptr(),                     (double*)dst->Mat.ptr,
                                       decx::utils::ceil<uint32_t>(src->Height(), 4) * 4,  dst->Pitch(), 
                                       &_transpose_config_2nd,                              &t1D);
    }
    else if constexpr (std::is_same_v<_type_out, uint8_t>) {
        decx::bp::transpose_8x8_caller((double*)planner.get_tmp1_ptr(),                                             (double*)dst->Mat.ptr,
                                       decx::utils::ceil<uint32_t>(src->Height(), _STG_alignment) * _STG_alignment, dst->Pitch(),
                                       &_transpose_config_2nd,                                                      &t1D);
    }
    else {
        decx::bp::transpose_4x4_caller((float*)planner.get_tmp1_ptr(),                                              (float*)dst->Mat.ptr,
                                       decx::utils::ceil<uint32_t>(src->Height(), _STG_alignment) * _STG_alignment, dst->Pitch(),
                                       &_transpose_config_2nd,                                                      &t1D);
    }
    planner.release_buffers();
}




_DECX_API_ de::DH de::dsp::cpu::FFT(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::FFT2D_caller_cplxf<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::fft::FFT2D_caller_cplxf<uint8_t>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::FFT2D_caller_cplxf<double>(_src, _dst, &handle);
        break;

    default:
        break;
    }

    return handle;
}



_DECX_API_ de::DH de::dsp::cpu::IFFT(de::Matrix& src, de::Matrix& dst, const de::_DATA_TYPES_FLAGS_ _output_type)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_output_type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::IFFT2D_caller_cplxf<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::fft::IFFT2D_caller_cplxf<uint8_t>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::IFFT2D_caller_cplxf<double>(_src, _dst, &handle);
        break;

    default:
        break;
    }

    return handle;
}