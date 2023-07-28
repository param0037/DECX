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


namespace decx
{
    namespace reduce
    {
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

    template <bool _print>
    void generate_configs(const uint64_t proc_len_v1, decx::cuda_stream* S, de::DH* handle);


    template <bool _print>
    void generate_configs(decx::PtrInfo<void> dev_src, const uint64_t proc_len_v1, decx::cuda_stream* S, de::DH* handle);

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
        void cuda_reduce1D_cmp_fp16_caller_Async(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S);


        template <bool _src_from_device>
        void cuda_reduce1D_sum_u8_i32_caller_Async(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S);


        template <bool _src_from_device, bool _is_max>
        void cuda_reduce1D_cmp_u8_caller_Async(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S);


        template <bool _src_from_device>
        void cuda_reduce1D_sum_fp16_fp32_caller_Async(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S);
    }
}


#endif