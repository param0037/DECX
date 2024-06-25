/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../FFT2D_kernels.cuh"


// [32 * 2, 8] = [64, 8]
__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R4_1st_R2C_uc8_cplxd(const uint8_t* __restrict src,
                                               double2* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc_v1,
                                               const uint32_t _pitchdst_v1)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 4;

    uint8_t recv[4];
    decx::utils::_cuda_vec128 res;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc_v1)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i] = src[_FFT_domain_dex * _pitchsrc_v1 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 4;

    if (tidy < _Bops_num && tidx < _pitchdst_v1)
    {
        // 1st
        res._vd = decx::utils::vec2_set1_fp64(0.f);
        res._cplxd.real = __dadd_rn(__dadd_rn(__ull2double_rn(recv[0]), __ull2double_rn(recv[1])),
                                    __dadd_rn(__ull2double_rn(recv[2]), __ull2double_rn(recv[3])));
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        ++_FFT_domain_dex;

        // 2nd
        res._cplxd.real = __dsub_rn(__ull2double_rn(recv[0]), __ull2double_rn(recv[2]));
        res._cplxd.image = __dsub_rn(__ull2double_rn(recv[1]), __ull2double_rn(recv[3]));
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        ++_FFT_domain_dex;

        // 3rd
        res._vd = decx::utils::vec2_set1_fp64(0.f);
        res._cplxd.real = __dadd_rn(__dsub_rn(__ull2double_rn(recv[0]), __ull2double_rn(recv[1])), 
                                    __dsub_rn(__ull2double_rn(recv[2]), __ull2double_rn(recv[3])));
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        ++_FFT_domain_dex;

        // 4th 
        res._cplxd.real = __dsub_rn(__ull2double_rn(recv[0]), __ull2double_rn(recv[2]));
        res._cplxd.image = __dsub_rn(__ull2double_rn(recv[3]), __ull2double_rn(recv[1]));
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
    }
}



__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R4_C2R_cplxd_u8(const double2* __restrict src,
                                              uint8_t* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc_v1,
                                              const uint32_t _pitchdst_v1)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 4;

    decx::utils::_cuda_vec128 recv[4], tmp;
    decx::utils::_cuda_vec128 tmp1, tmp2;
    decx::utils::_cuda_vec64 res;
    de::CPd W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc_v1)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vd = src[_FFT_domain_dex * _pitchsrc_v1 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W.construct_with_phase(__dmul_rn(Two_Pi, __ddiv_rn(__ull2double_rn(_warp_loc_id), 
                                                       __ull2double_rn(_kernel_info._warp_proc_len))));
    recv[1]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[1]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Four_Pi, __ddiv_rn(__ull2double_rn(_warp_loc_id),
                                                        __ull2double_rn(_kernel_info._warp_proc_len))));
    recv[2]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[2]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Six_Pi, __ddiv_rn(__ull2double_rn(_warp_loc_id), 
                                                       __ull2double_rn(_kernel_info._warp_proc_len))));
    recv[3]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[3]._cplxd, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v1)
    {
        // Calculate the first and third output
        tmp1._vd = decx::utils::cuda::__double_add2(recv[0]._vd, recv[2]._vd);
        tmp2._vd = decx::utils::cuda::__double_add2(recv[1]._vd, recv[3]._vd);

        // Store the first output
        res._fp64 = __dadd_rn(tmp1._vd.x, tmp2._vd.x);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = __double2ull_rn(res._fp64);
        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        
        // Store the third output
        res._fp64 = __dsub_rn(tmp1._vd.x, tmp2._vd.x);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = __double2ull_rn(res._fp64);

        // Calculate and store the second output
        res._fp64 = __fsub_rn(__dadd_rn(recv[0]._cplxd.real, recv[3]._cplxd.image), 
                              __dadd_rn(recv[1]._cplxd.image, recv[2]._cplxd.real));
        _FFT_domain_dex -= (_kernel_info._store_pitch);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = __double2ull_rn(res._fp64);

        res._fp64 = __fsub_rn(__dadd_rn(recv[0]._cplxd.real, recv[1]._cplxd.image), 
                               __dadd_rn(recv[2]._cplxd.real, recv[3]._cplxd.image));

        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = __double2ull_rn(res._fp64);
    }
}
