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


#define _CU_FILTER2D_FP32_BLOCK_X_ _WARP_SIZE_
#define _CU_FILTER2D_FP32_BLOCK_Y_ 8


namespace decx
{
    namespace dsp {
        template <typename _data_type>
        class cuda_Filter2D_planner;
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

public:
    cuda_Filter2D_planner() {}


    void _CRSR_ plan(const decx::_matrix_layout* src_layout, const decx::_matrix_layout* _kernel_layout,
        const de::extend_label _method, decx::cuda_stream* S, de::DH* handle);


    uint2 dst_dims_req() const;


    void _CRSR_ run(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst,
        decx::cuda_stream* S, de::DH* handle);
};


#endif
