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

#ifndef _CPU_GENERATOR_H_
#define _CPU_GENERATOR_H_

#include <Element_wise/common/cpu_element_wise_planner.h>
#include <thread_management/thread_arrange.h>
#include <Classes/Matrix.h>
#include <Classes/Number.h>


namespace de
{
namespace cpu
{
    _DECX_API_ void Generate(de::OutputMatrix mat, const de::_DATA_TYPES_FLAGS_ type_out, const de::Number val = de::Number());


    _DECX_API_ void Random(de::OutputMatrix mat, const de::_DATA_TYPES_FLAGS_ type_out, const uint32_t width, const uint32_t height, const int32_t seed, const de::Point2D_d range);
}
}


namespace decx
{
    class cpu_Generator2D;
}

class decx::cpu_Generator2D : public decx::cpu_ElementWise2D_planner
{
public:
    template <typename FuncType, typename _data_type, class ...Args>
    void fill_caller(FuncType&& f, _data_type* buf, const uint32_t pitch_v1, decx::utils::_thr_1D *t1D, Args&& ...additional)
    {
        _data_type* loc_ptr = buf;

        uint32_t _thr_cnt = 0;

        for (int32_t i = 0; i < this->_thread_dist.y; ++i)
        {
            loc_ptr = buf + pitch_v1 * i * this->_fmgr_WH[1].frag_len;
            for (int32_t j = 0; j < this->_thread_dist.x; ++j){
                uint2 proc_dims = 
                    make_uint2(j < this->_thread_dist.x - 1 ? this->_fmgr_WH[0].frag_len : this->_fmgr_WH[0].last_frag_len,
                            i < this->_thread_dist.y - 1 ? this->_fmgr_WH[1].frag_len : this->_fmgr_WH[1].last_frag_len);

                t1D->_async_thread[_thr_cnt] = decx::cpu::register_task_default(f, loc_ptr, proc_dims, pitch_v1, additional...);
                
                loc_ptr += this->_fmgr_WH[0].frag_len;
                ++_thr_cnt;
            }
        }
        t1D->__sync_all_threads(make_uint2(0, _thr_cnt));
    }
};

#endif
