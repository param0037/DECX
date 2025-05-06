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

#include "../QR_factorization.h"
#include "Basic_process/transpose/CPU/transpose2D_config.h"
#include "../../../../core/thread_management/thread_arrange.h"

static decx::Ptr2D_Info<float> z;

static void Col_HouseHolder_proc_fp32(const float* p_col, const uint32_t len, const uint32_t local_idx, float* V_col)
{
    float sum = 0, norm2_post_x0 = 0;
    for (int i = local_idx; i < len; ++i){
        const float x = p_col[i];
        sum += x;
        if (i > local_idx)
            norm2_post_x0 += x * x;
    }
    if (fabs(sum) > 1e-3){
        const float x0 = p_col[local_idx];
        const float sign = (x0 < 0) ? 1 : -1;
        V_col[local_idx] = x0 - sign * sqrt(norm2_post_x0 + x0 * x0);
        
        // normalize
        const float norm2_v = sqrt(norm2_post_x0 + V_col[local_idx] * V_col[local_idx]);
        for (int i = local_idx; i < len; ++i){
            V_col[local_idx] = V_col[local_idx] / norm2_v;
        }
    }
}

// Transposed, col is row, row is col
static void Blocked_QR_HouseHolder_fp32(const float* src, const uint2 panel_dims, const uint32_t panel_pitch,
    float* V, float* W, uint2* res_dims, const uint32_t pitch_IWY)
{
    for (int i = 0; i < panel_dims.x; ++i){
        Col_HouseHolder_proc_fp32(src + i * panel_pitch, panel_dims.y, i, V + i * panel_pitch);
        // Calculate W
        if (i == 0){
            for (int k = 0; k < panel_dims.y; ++k){
                W[k] = V[k] * 2.0;
            }
            // Make z an I matrix
            for (int j = 0; j < panel_dims.y; ++j){
                z._ptr.ptr[j * z._dims.x + j] = 1.0;
            }
        }
        else{
            const float* p_V_col_last = V + (i - 1) * panel_pitch;
            float* p_W_col = W + i * panel_pitch;
            const float* p_V_col = p_V_col_last + panel_pitch;

            for (int j = 0; j < panel_dims.y; ++j){
                float* p_z_col = z._ptr.ptr + j * z._dims.x;
                const float multiplier = p_V_col_last[j];
                for (int k = 0; k < panel_dims.y; ++k){
                    p_z_col[k] -= multiplier * p_z_col[k];
                    p_W_col[k] = 2.0 * p_z_col[k] * p_V_col[k];
                }
            }
        }
    }
}

_DECX_API_ void de::blas::cpu::GQRF(de::Matrix& src, de::Matrix& Q, de::Matrix& R)
{
    de::DH* handle = de::GetLastError();

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _Q = dynamic_cast<decx::_Matrix*>(&Q);
    decx::_Matrix* _R = dynamic_cast<decx::_Matrix*>(&R);

    const uint32_t src_pitch = decx::utils::align<uint32_t>(_src->Height(), 8);
    const uint2 src_dims = make_uint2(_src->Height(), _src->Width());
    decx::PtrInfo<float> buffer;
    decx::alloc::_host_virtual_page_malloc(&buffer, src_pitch * src_dims.y * sizeof(float));

    const uint32_t block_dim = 4;

    const uint2 z_dims = make_uint2(decx::utils::align<uint32_t>(_src->Width(), 8), _src->Width());
    decx::alloc::_host_virtual_page_malloc(&z._ptr, z_dims.x * z_dims.y * sizeof(float));
    z._dims = z_dims;

    // Transpose and store to buffer
    decx::blas::_cpu_transpose_config tp_config;
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());
    tp_config.config(sizeof(float), decx::cpu::_get_permitted_concurrency(), src_dims, handle);
    tp_config.transpose_4b_caller((float*)_src->Mat.ptr, buffer.ptr, _src->Pitch(), src_pitch, &t1D);

    for (int32_t i = 0; i < src_dims.x / block_dim; ++i)
    {

    }

    decx::alloc::_host_virtual_page_dealloc(&buffer);
}