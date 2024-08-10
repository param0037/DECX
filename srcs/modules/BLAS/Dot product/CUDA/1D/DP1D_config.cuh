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


#ifndef _DP1D_CONFIG_CUH_
#define _DP1D_CONFIG_CUH_


#include "../DP_kernels.cuh"
#include "../../../../../common/Algorithms/Reduce_CUDA/reduce_sum.cuh"
#include "../../../../../common/Algorithms/Reduce_CUDA/reduce_callers.cuh"
#include "../../../../core/allocators.h"


namespace decx
{
    namespace blas
    {
        template <typename _type_in>
        class cuda_DP1D_configs;
    }
}



template <typename _type_in>
class decx::blas::cuda_DP1D_configs
{
private:
    void* _post_proc_conf;

    /**
    * Change of load byte, for example :
    *   uint8_t -> uint32_t
    *   fp16 -> fp32
    */
    bool _load_byte_changed;


    uint64_t _proc_len_v1;

    uint64_t _grid_len_k1;

    /**
    * If the vector length is too large to cover for the thread block, then post-processing reduction is needed
    * @return : True if post-processing is needed; Otherwise, False
    */
    bool _post_proc_needed;

    // Are dev_A and dev_B from the on-deivce classes
    bool _from_dev;

public:
    decx::PtrInfo<void> _dev_A, _dev_B, _dev_dst;

    cuda_DP1D_configs();


    cuda_DP1D_configs(const uint64_t _proc_len, decx::cuda_stream* S, const uint32_t _fp16_accu = 0);


    cuda_DP1D_configs(decx::PtrInfo<void> dev_A, decx::PtrInfo<void> dev_B, const uint64_t _proc_len,
        decx::cuda_stream* S, const uint32_t _fp16_accu = 0);

    /**
    * If the vector length is too large to cover for the thread block, then post-processing reduction is needed
    * @return : True if post-processing is needed; Otherwise, False
    */
    bool postproc_needed() const;


    uint64_t get_actual_proc_len() const;


    uint64_t get_grid_len_k1() const;


    void relase_buffer();

    template <typename _config_type>
    decx::reduce::cuda_reduce1D_configs<_config_type>* get_configs_ptr();


    ~cuda_DP1D_configs();
};


#endif
