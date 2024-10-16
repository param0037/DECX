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

#include <Classes/Matrix.h>
#include "bisection/CPU/eig_bisect.h"


namespace de
{
namespace blas{
namespace cpu{
    _DECX_API_ void Eigenvalue(de::InputMatrix src, float** a, float** b);
}
}
}

_DECX_API_ void de::blas::cpu::Eigenvalue(de::InputMatrix src, float** a, float** b)
{
    de::ResetLastError();

    const decx::_Matrix* _src = dynamic_cast<const decx::_Matrix*>(&src);

    decx::blas::cpu_eig_bisection<float> planner;
    const uint32_t conc = decx::cpu::_get_permitted_concurrency();
    planner.Init(conc, &_src->get_layout(), 0.001, de::GetLastError());

    decx::utils::_thread_arrange_1D t1D(conc);

    // planner.extract_diagonal((float*)_src->Mat.ptr, &t1D);

    *a = planner.get_diag();
    *b = planner.get_off_diag();

    // planner.calc_Gerschgorin_bound(&t1D);

    planner.plan(_src, &t1D, de::GetLastError());
    printf("bound : (%f, %f)\n", planner.get_Gerschgorin_L() , planner.get_Gerschgorin_U());

    clock_t s, e;
    s = clock();
    // for (int i = 0; i < 1000; ++i) {
        planner.iter_bisection();
        // planner.reset();
    // }
    e = clock();


    auto* read_buf = planner.get_valid_intervals_array();
    printf("planner.get_eig_count() : %d\n", planner.get_eig_count());
    int32_t id = 0;
    for (int i = 0; i < planner.get_eig_count(); ++i){
        if (read_buf[i].is_valid()) {
            printf("[%d] (%f, %f), count(LU) = %d, %d\n", id, read_buf[i]._l, read_buf[i]._u, read_buf[i]._count_l, read_buf[i]._count_u);
            ++id;
        }
    }

    printf("time spent msec : %lf\n", (double)(e - s) / (double)CLOCKS_PER_SEC * 1000 / 1);
}
