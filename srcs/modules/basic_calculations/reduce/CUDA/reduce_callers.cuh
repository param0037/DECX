/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _REDUCE_CALLERS_H_
#define _REDUCE_CALLERS_H_


#include "reduce_sum.cuh"
#include "reduce_cmp.cuh"
#include "../../../classes/classes_util.h"


/**
* To perform parallel reduction, ping-pong buffers are required when there are
* multiple blocks at a reduction kernel. Since the reduction can only been done
* within a block, when a data array is too long to be covered by a single block,
* multiple results will be generated after launching the kernel. Hence, another
* reduction kernel should be lauched to calculate the reduction result on the previous
* results. That is the main reason why ping-pong memory is needed.
*
* Stage 0 refers to the stage when a reduction kernel is firstly launch.
* Stage 1 refers to the stage when a reduction kernel launches for the second time. Its
* required processing length is reducing. The same to stage n.
*
* To save the memory resources, buffer 2 will not be allocated with the memory size same
* as source data array. Instead, it will be <length / block_process_length>, which is
* indicated as stage 1 length. The same as stage n length.
*/


namespace decx
{
    namespace reduce
    {
        /**
        * This class manages the behaviours of parallel reduction and stores the key parameters
        * needed for the CUDA kernel arrangments.
        */
        template <typename _type_in>
        class cuda_reduce1D_configs;
    }
}



template <typename _type_in>
class decx::reduce::cuda_reduce1D_configs
{
private:
    decx::PtrInfo<void> _dev_src;
    decx::PtrInfo<void> _d_tmp1, _d_tmp2;

    _type_in _fill_val;

    uint64_t _proc_len;
    uint64_t _actual_len;

    decx::alloc::MIF<void> _MIF_tmp1, _MIF_tmp2;

public:
    cuda_reduce1D_configs() {}

    /**
    * This generation is especially for the case when the source memory is located at host, which requires
    * memcpy from host to device.
    * In this case, same size as source memory will be allocated to buffer 1 and stage 1 of 1D process length
    * will be allocated to buffer 2.
    *
    * @param dev_src : The source memory referred to device memory
    * @param proc_len_v1 : The actual reduce length
    * @param S : The pointer of cuda-stream
    */
    void generate_configs(const uint64_t proc_len_v1, decx::cuda_stream* S);


    /**
    * This generation is especially for the case when the source memory is located at device.
    * In this case, same size of memory with stage 1 1D process length will be allocated to buffer 1 and 2
    * Buffer 1 will be set as leading first.
    * 
    * @param dev_src : The source memory referred to device memory
    * @param proc_len_v1 : The actual reduce length
    * @param S : The pointer of cuda-stream
    */
    void generate_configs(decx::PtrInfo<void> dev_src, const uint64_t proc_len_v1, decx::cuda_stream* S);


    uint64_t get_proc_len() const;
    uint64_t get_actual_len() const;


    void set_fill_val(const _type_in _val);


    decx::PtrInfo<void> get_dev_src() const;
    decx::PtrInfo<void> get_dev_tmp1() const;
    decx::PtrInfo<void> get_dev_tmp2() const;

    _type_in get_fill_val() const;

    void inverse_mutex_MIF_states();

    decx::alloc::MIF<void> get_leading_MIF() const;
    decx::alloc::MIF<void> get_lagging_MIF() const;


    void release_buffer();
};


namespace decx
{
    namespace reduce
    {
        template <bool _src_from_device>
        void cuda_reduce1D_sum_fp32_caller_Async(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, decx::cuda_stream* S);


        template <bool _src_from_device, bool _is_max>
        void cuda_reduce1D_cmp_fp32_caller_Async(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, decx::cuda_stream* S);


        template <bool _src_from_device, bool _is_max>
        void cuda_reduce1D_cmp_fp64_caller_Async(decx::reduce::cuda_reduce1D_configs<double>* _kp_configs, decx::cuda_stream* S);



        template <bool _src_from_device, bool _is_max>
        void cuda_reduce1D_cmp_fp16_caller_Async(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S);


        template <bool _src_from_device>
        void cuda_reduce1D_sum_u8_i32_caller_Async(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S);


        template <bool _src_from_device, bool _is_max>
        void cuda_reduce1D_cmp_u8_caller_Async(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S);


        template <bool _src_from_device>
        void cuda_reduce1D_sum_fp16_fp32_caller_Async(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S);
    }
}


namespace decx
{
    namespace reduce
    {
        template <typename _type_in>
        class cuda_reduce2D_1way_configs;
    }
}


template <typename _type_in>
class decx::reduce::cuda_reduce2D_1way_configs
{
private:
    decx::PtrInfo<void> _dev_src;
    decx::Ptr2D_Info<void> _d_tmp1, _d_tmp2;

    uint2 _proc_dims_actual;
    uint2 _proc_dims_v;

    uint32_t _kernel_call_times;

    decx::alloc::MIF<void> _MIF_tmp1, _MIF_tmp2;


    uint32_t _calc_reduce_kernel_call_times() const;


public:
    cuda_reduce2D_1way_configs() {}

    template <bool _is_reduce_h>
    void generate_configs(const uint2 proc_dims, decx::cuda_stream* S);


    decx::Ptr2D_Info<void> get_dtmp1() const;
    decx::Ptr2D_Info<void> get_dtmp2() const;


    uint2 get_actual_proc_dims() const;


    uint2 get_proc_dims_v() const;


    void* get_leading_ptr() const;
    void* get_lagging_ptr() const;


    void reverse_MIF_states();

    uint32_t get_kernel_call_times() const;


    void release_buffer();
};


namespace decx
{
    namespace reduce
    {
        void reduce_sum2D_h_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);


        void reduce_sum2D_v_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);


        template <bool _src_from_device>
        void reduce_sum2D_full_fp32_Async(decx::reduce::cuda_reduce1D_configs<float>* _configs, const uint2 proc_dims, 
            const uint32_t _pitch_src_v4, decx::cuda_stream* S);


        template <bool _is_max>
        void reduce_cmp2D_h_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);

        template <bool _is_max>
        void reduce_cmp2D_v_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);


        template <bool _is_max, bool _src_from_device>
        void reduce_cmp2D_full_fp32_Async(decx::reduce::cuda_reduce1D_configs<float>* _configs, const uint2 proc_dims,
            const uint32_t _pitch_src_v4, decx::cuda_stream* S);
    }
}


#endif