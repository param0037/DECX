/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#include "../FFT2D_config.cuh"
#include "../../../../../../common/double_buffer.h"
#include "../../../../../../common/Basic_process/transpose/CUDA/transpose_kernels.cuh"
#include "../FFT2D_1way_kernel_callers.cuh"



// __global__
// void cu_tr_b8(const double2* src,
//               double2 *dst, 
//               const uint32_t pitchsrc_v2,        // in double2 (de::CPf x2)
//               const uint32_t pitchdst_v2,        // in double2 (de::CPf x2)
//               const uint2 dst_dims)   // in de::CPf
// {
//     uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
//     uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

//     uint64_t dex = (tidy << 3) * pitchsrc_v2 + tidx;

//     __shared__ double2 _buffer[64][32 + 1];

//     double2 _regs[8], _transp[4];
//     for (uint8_t i = 0; i < 8; ++i) {
//         _regs[i] = decx::utils::vec2_set1_fp64(0.0);
//     }
//     // Load source data to registers
//     if (tidx < pitchsrc_v2) {
//         for (uint8_t i = 0; i < 8; ++i) {
//             if ((tidy << 3) + i < dst_dims.x) { _regs[i] = src[dex + pitchsrc_v2 * i]; }
//         }
//     }
//     // On-register transpose (lane 0)
//     _transp[0].x = _regs[0].x;      _transp[0].y = _regs[1].x;
//     _transp[1].x = _regs[2].x;      _transp[1].y = _regs[3].x;
//     _transp[2].x = _regs[4].x;      _transp[2].y = _regs[5].x;
//     _transp[3].x = _regs[6].x;      _transp[3].y = _regs[7].x;
//     // Store to shared memory, also in transposed form (lane 0)
//     for (uint8_t i = 0; i < 4; ++i) {
//         _buffer[threadIdx.x * 2][threadIdx.y * 4 + i] = _transp[i];
//     }
//     // On-register transpose (lane 1)
//     _transp[0].x = _regs[0].y;      _transp[0].y = _regs[1].y;
//     _transp[1].x = _regs[2].y;      _transp[1].y = _regs[3].y;
//     _transp[2].x = _regs[4].y;      _transp[2].y = _regs[5].y;
//     _transp[3].x = _regs[6].y;      _transp[3].y = _regs[7].y;
//     // Store to shared memory, also in transposed form (lane 1)
//     for (uint8_t i = 0; i < 4; ++i) {
//         _buffer[threadIdx.x * 2 + 1][threadIdx.y * 4 + i] = _transp[i];
//     }

//     __syncthreads();

//     for (uint8_t i = 0; i < 8; ++i) {
//         _regs[i] = _buffer[threadIdx.y * 8 + i][threadIdx.x];
//     }

//     tidx = threadIdx.x + blockIdx.y * blockDim.x;
//     tidy = threadIdx.y + blockIdx.x * blockDim.y;

//     dex = tidy * 8 * pitchdst_v2 + tidx;

//     if (tidx < pitchdst_v2) {
//         for (uint8_t i = 0; i < 8; ++i) {
//             if (tidy * 8 + i < dst_dims.y) { dst[dex + pitchdst_v2 * i] = _regs[i]; }
//         }
//     }
// }



// static void 
// tr_b8(const double2* src, 
//                          double2* dst, 
//                          const uint2 proc_dims_dst,
//                          const uint32_t pitchsrc, 
//                          const uint32_t pitchdst, 
//                          decx::cuda_stream* S)
// {
//     dim3 transp_thread_0(32, 8);
//     dim3 transp_grid_0(decx::utils::ceil<uint>(proc_dims_dst.y, 64),
//         decx::utils::ceil<uint>(proc_dims_dst.x, 64));
//     printf("trabspose transp_grid_0.x : %d, transp_grid_0.y : %d\n", transp_grid_0.x, transp_grid_0.y);
//     cu_tr_b8 << <transp_grid_0, transp_thread_0, 0, S->get_raw_stream_ref() >> > (
//         src, dst, pitchsrc / 2, pitchdst / 2, proc_dims_dst);
// }


template <>
template <typename _type_in>
void decx::dsp::fft::_cuda_FFT2D_planner<float>::Forward(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S) const
{
    decx::utils::double_buffer_manager double_buffer(this->get_tmp1_ptr<void>(), this->get_tmp2_ptr<void>());

    decx::dsp::fft::FFT2D_cplxf_1st_1way_caller<_type_in, false>(src->Mat.ptr, &double_buffer,
        this->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S);
    
    decx::blas::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                             double_buffer.get_lagging_ptr<double2>(),
                             make_uint2(this->get_buffer_dims().y, this->get_buffer_dims().x),
                             this->get_buffer_dims().x, 
                             this->get_buffer_dims().y, 
                             S);
    
    double_buffer.update_states();

    decx::dsp::fft::FFT2D_C2C_cplxf_1way_caller<_FFT2D_END_(de::CPf)>(&double_buffer,
        this->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);

    decx::blas::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                             (double2*)dst->Mat.ptr,
                             make_uint2(dst->Width(), dst->Height()),
                             this->get_buffer_dims().y, 
                             dst->Pitch(), S);
}

template void decx::dsp::fft::_cuda_FFT2D_planner<float>::Forward<float>(decx::_GPU_Matrix*, decx::_GPU_Matrix*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT2D_planner<float>::Forward<de::CPf>(decx::_GPU_Matrix*, decx::_GPU_Matrix*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT2D_planner<float>::Forward<uint8_t>(decx::_GPU_Matrix*, decx::_GPU_Matrix*, decx::cuda_stream*) const;



template <>
template <typename _type_out>
void decx::dsp::fft::_cuda_FFT2D_planner<float>::Inverse(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S) const
{
    decx::utils::double_buffer_manager double_buffer(this->get_tmp1_ptr<void>(), this->get_tmp2_ptr<void>());
    
    decx::dsp::fft::FFT2D_cplxf_1st_1way_caller<de::CPf, true>(src->Mat.ptr, &double_buffer,
        this->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S);

    decx::blas::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                             double_buffer.get_lagging_ptr<double2>(),
                             make_uint2(this->get_buffer_dims().y, this->get_buffer_dims().x),
                             this->get_buffer_dims().x, this->get_buffer_dims().y, 
                             S);
    double_buffer.update_states();

    decx::dsp::fft::FFT2D_C2C_cplxf_1way_caller<_IFFT2D_END_(_type_out)>(&double_buffer,
        this->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);
    if (std::is_same<_type_out, de::CPf>::value) {
        decx::blas::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                                 (double2*)dst->Mat.ptr,
                                 make_uint2(dst->Width(), dst->Height()),
                                 this->get_buffer_dims().y, 
                                 dst->Pitch(), S);
    }
    else if (std::is_same<_type_out, uint8_t>::value) {
        decx::blas::transpose2D_b1(double_buffer.get_leading_ptr<uint32_t>(), 
                                 (uint32_t*)dst->Mat.ptr,
                                 make_uint2(dst->Width(), dst->Height()),
                                 this->get_buffer_dims().y * 8,  // Times 8 cuz 8 uchars in one de::CPf
                                 dst->Pitch(), S);
    }
    else if (std::is_same<_type_out, float>::value) {
        decx::blas::transpose2D_b4(double_buffer.get_leading_ptr<float2>(), 
                                 (float2*)dst->Mat.ptr,
                                 make_uint2(dst->Width(), dst->Height()),
                                 this->get_buffer_dims().y * 2,  // Times 2 cuz 2 floats in one de::CPf
                                 dst->Pitch(), S);
    }
}

template void decx::dsp::fft::_cuda_FFT2D_planner<float>::Inverse<float>(decx::_GPU_Matrix*, decx::_GPU_Matrix*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT2D_planner<float>::Inverse<de::CPf>(decx::_GPU_Matrix*, decx::_GPU_Matrix*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT2D_planner<float>::Inverse<uint8_t>(decx::_GPU_Matrix*, decx::_GPU_Matrix*, decx::cuda_stream*) const;


decx::ResourceHandle decx::dsp::fft::cuda_FFT2D_cplxf32_planner;
decx::ResourceHandle decx::dsp::fft::cuda_IFFT2D_cplxf32_planner;
