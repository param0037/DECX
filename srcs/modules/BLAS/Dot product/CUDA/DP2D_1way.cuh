/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _DP2D_1WAY_CUH_
#define _DP2D_1WAY_CUH_


#include "DP_kernels.cuh"
#include "../../../basic_calculations/reduce/CUDA/reduce_callers.cuh"
#include "../../../basic_calculations/reduce/CUDA/reduce_sum.cuh"
#include "../../../basic_calculations/reduce/CUDA/reduce_callers.cuh"
#include "../../../core/allocators.h"
#include "../../../classes/Matrix.h"
#include "../../../classes/Vector.h"


namespace decx
{
    namespace blas
    {
        template <typename _type_in>
        class cuda_DP2D_configs;
    }
}



template <typename _type_in>
class decx::blas::cuda_DP2D_configs
{
private:
    void* _post_proc_conf;

    /**
    * Change of load byte, for example :
    *   uint8_t -> uint32_t
    *   fp16 -> fp32
    */
    bool _load_byte_changed;


    uint2 _proc_dims;

    /**
    * If the vector length is too large to cover for the thread block, then post-processing reduction is needed
    * @return : True if post-processing is needed; Otherwise, False
    */
    bool _post_proc_needed;

    // Are dev_A and dev_B from the on-deivce classes
    bool _from_dev;


    uint32_t _grid_proc_dim_k1;


    dim3 _first_kernel_config;

public:
    decx::PtrInfo<void> _dev_A, _dev_B, _dev_dst;


    uint2 _dev_mat_dims;


    cuda_DP2D_configs();


    template <bool _is_reduce_h>
    void generate_config(const uint2 proc_dims, decx::cuda_stream* S, const uint32_t _fp16_accu = 0);


    /*template <bool _is_reduce_h>
    void generate_config(decx::PtrInfo<void> dev_A, decx::PtrInfo<void> dev_B, const uint2 proc_dims,
        decx::cuda_stream* S, const uint32_t _fp16_accu = 0);*/


    template <bool _is_reduce_h>
    void alloc_buffers(decx::cuda_stream* S, const uint32_t _fp16_accu);


    void release_buffer();


    template <typename _config_type>
    decx::reduce::cuda_reduce2D_1way_configs<_config_type>* get_configs_ptr();


    uint2 get_actual_proc_dims() const;


    dim3 get_1st_kernel_config() const;


    bool postproc_needed() const;
};



#endif