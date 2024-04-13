/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CUDA_FILTER2D_PLANNER_CUH_
#define _CUDA_FILTER2D_PLANNER_CUH_


#include "../../../../core/basic.h"
#include "../../../../classes/GPU_Matrix.h"
#include "../../../../BLAS/basic_process/extension/extend_flags.h"
#include "../../../../core/resources_manager/decx_resource.h"


#define _CU_FILTER2D_FP32_BLOCK_X_ _WARP_SIZE_
#define _CU_FILTER2D_FP32_BLOCK_Y_ 8


namespace decx
{
    namespace dsp 
    {
        template <typename _data_type>
        class cuda_Filter2D_planner;


        typedef void(*_cu_F2_U8_Kcaller) (const decx::dsp::cuda_Filter2D_planner<uint8_t>*, const double*,
            const void*, void*, const uint32_t, decx::cuda_stream*);
        typedef void(*_cu_F2_FP32_Kcaller) (const decx::dsp::cuda_Filter2D_planner<float>*, const float4*,
            const float*, float4*, const uint32_t, decx::cuda_stream*);

        extern decx::dsp::_cu_F2_U8_Kcaller _cu_F2_U8_Kcallers[2][32];


        extern decx::dsp::_cu_F2_FP32_Kcaller _cu_F2_FP32_Kcallers[32];


        /*extern decx::dsp::cuda_Filter2D_planner<float>* _cuda_filter2D_fp32;
        extern decx::dsp::cuda_Filter2D_planner<uint8_t>* _cuda_filter2D_u8;*/

        extern decx::ResourceHandle _cuda_filter2D_fp32;
        extern decx::ResourceHandle _cuda_filter2D_u8;
    }
}


template <typename _data_type>
class decx::dsp::cuda_Filter2D_planner
{
private:
    const decx::_matrix_layout* _src_layout;
    const decx::_matrix_layout* _kernel_layout;

    uint2 _dst_dims;

    de::extend_label _conv_border_method;

    decx::Ptr2D_Info<void> _ext_src;

    dim3 _block, _grid;

    de::_DATA_TYPES_FLAGS_ _output_type;

public:
    template <uint32_t _ext_w>
    static void _cu_Filter2D_fp32_caller(const decx::dsp::cuda_Filter2D_planner<float>* _fake_this, const float4* src,
        const float* kernel, float4* dst, const uint32_t pitchdst_v1, decx::cuda_stream* S);


    template <uint32_t _ext_w>
    static void _cu_Filter2D_NB_u8_x_caller(const decx::dsp::cuda_Filter2D_planner<uint8_t>* _fake_this, const double* src, 
        const void* kernel, void* dst, const uint32_t pitchdst_v1, decx::cuda_stream* S);


    template <uint32_t _ext_w>
    static void _cu_Filter2D_BC_u8_x_caller(const decx::dsp::cuda_Filter2D_planner<uint8_t>* _fake_this, const double* src, 
        const void* kernel, void* dst, const uint32_t pitchdst_v1, decx::cuda_stream* S);


public:
    cuda_Filter2D_planner();


    bool changed(const decx::_matrix_layout* src_layout, const decx::_matrix_layout* kernel_layout, const de::extend_label extend_method,
        const de::_DATA_TYPES_FLAGS_ output_type) const;


    void _CRSR_ plan(const decx::_matrix_layout* src_layout, const decx::_matrix_layout* _kernel_layout,
        const de::extend_label _method, const de::_DATA_TYPES_FLAGS_ output_type, decx::cuda_stream* S, de::DH* handle);


    uint2 dst_dims_req() const;


    void _CRSR_ run(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst,
        decx::cuda_stream* S, de::DH* handle);


    static bool validate_kerW(const uint32_t kerW);


    static void release(decx::dsp::cuda_Filter2D_planner<_data_type>* _fake_this);


    ~cuda_Filter2D_planner();
};


namespace decx
{
    namespace dsp {
        void Release_CU_Filter2D_Resource();
    }
}


#endif
