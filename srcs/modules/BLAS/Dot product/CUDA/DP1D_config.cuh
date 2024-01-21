/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _DP1D_CONFIG_CUH_
#define _DP1D_CONFIG_CUH_


#include "DP_kernels.cuh"
#include "../../../basic_calculations/reduce/CUDA/reduce_sum.cuh"
#include "../../../basic_calculations/reduce/CUDA/reduce_callers.cuh"
#include "../../../core/allocators.h"


namespace decx
{
    namespace dot
    {
        template <typename _type_in>
        class cuda_DP1D_configs;
    }
}



template <typename _type_in>
class decx::dot::cuda_DP1D_configs
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