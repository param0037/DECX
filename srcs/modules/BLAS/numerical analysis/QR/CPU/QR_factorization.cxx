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
#include "householder_reflector.h"
#include "blocked_GQR_planner.h"


static decx::Ptr2D_Info<float> z;


// Transposed, col is row, row is col
static void Blocked_QR_HouseHolder_fp32(const float* src, const uint2 panel_dims, const uint32_t panel_pitch,
    float* V, float* W, uint2* res_dims, const uint32_t pitch_IWY)
{
    for (int i = 0; i < panel_dims.x; ++i){
        decx::blas::householder_calc_v8_fp32(src + ((i >> 3) << 3), V + ((i >> 3) << 3), panel_dims.y - i, i);
    }
}

_DECX_API_ void de::blas::cpu::GQRF(de::Matrix& src, de::Matrix& Q, de::Matrix& R)
{
    de::DH* handle = de::GetLastError();

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _Q = dynamic_cast<decx::_Matrix*>(&Q);
    decx::_Matrix* _R = dynamic_cast<decx::_Matrix*>(&R);

    decx::blas::Blocked_GQR_planner<float> _planner;

    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    _planner.Config(make_uint2(3, _src->Height()), handle);
    
    for (int i = 0; i < _src->Width() / 3; ++i) {
        if (i == 0) {
            // Flush buffers
            _planner.FlushAllTiles();

            // // Load to panel
            _planner.LoadSrcTile(_src->Mat.GetRawPtr<float>(), i, _src->Pitch(), &t1D);

            // // Calculate block householder
            _planner.Process_HouseHolder();
        }
    }

    const float* V = _planner.GetV();
    // const float* V = _planner.GetTile();
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < _src->Height(); ++i) {
            printf("%f, ", V[j * 8 + i]);
        }
        printf("\n");
    }
}