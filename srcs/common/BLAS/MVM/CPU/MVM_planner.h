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

#ifndef MVM_PLANNER_H
#define MVM_PLANNER_H

#include <basic.h>
#include <vector_defines.h>
#include <FMGR/fragment_arrangment.h>
#include <PtrInfo.h>
#include <Element_wise/common/cpu_element_wise_planner.h>
#include <thread_management/thread_pool.h>


namespace decx
{
namespace blas
{
    template <typename _data_type>
    class cpu_MVM_planner;

}
}

template <typename _data_type>
class decx::blas::cpu_MVM_planner
{
private:
    uint2 _mat_dims;
    decx::utils::frag_manager _fmgr_H;
    decx::utils::frag_manager _fmgr_L;

    uint32_t _alignment;
    uint32_t _concurrency;

    decx::PtrInfo<decx::utils::frag_manager> _block_confs_H_perthread;
    uint8_t _L_align_mask[256];


    void AlignMaskGen(const uint32_t width_leftover, const uint32_t simd_align_byte);

public:
    cpu_MVM_planner();
    ~cpu_MVM_planner();

    int32_t Config(const uint2 mat_dims);
    
    /**
     * @brief Performing alpha .* mat * vec + c
     */
    int32_t Run(const _data_type* __restrict mat, const _data_type* __restrict vec, _data_type* __restrict res_vec, const uint32_t mat_pitch,
        const _data_type alpha = _data_type(1), const _data_type beta = _data_type(0));


    const decx::utils::frag_manager* GetFmgrL() const {
        return &this->_fmgr_L;
    }


    const decx::utils::frag_manager* GetFmgrH() const {
        return &this->_fmgr_H;
    }


    static int32_t Release(cpu_MVM_planner<_data_type>* _fake_this);
};

#endif