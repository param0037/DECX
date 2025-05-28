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


#ifndef _REDUCE_CALLERS_CUH_
#define _REDUCE_CALLERS_CUH_


#include "reduce_sum.cuh"
#include "reduce_cmp.cuh"
#include <Classes/classes_util.h>


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



#define _CU_REDUCE1D_MEM_ALIGN_8B_ 2
#define _CU_REDUCE1D_MEM_ALIGN_4B_ 4
#define _CU_REDUCE1D_MEM_ALIGN_2B_ 8
#define _CU_REDUCE1D_MEM_ALIGN_1B_ 16



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


        template <typename _Ty>
        struct cu_reduce1D_param_pack;


        template <typename _Ty>
        using RWPK_1D = decx::reduce::cu_reduce1D_param_pack<_Ty>;


        /**
        * @return True if the flatten length is more than 1; False if the flatten length is 1 (Only flatten_lernel is called)
        */
        template <typename _type_in, typename _type_postproc>
        bool reduce2D_flatten_postproc_configs_gen(decx::reduce::cuda_reduce1D_configs<_type_postproc>* _configs_ptr,
            const uint32_t pirchsrc, const uint2 proc_dims_v1, decx::cuda_stream* S);
    }
}



namespace decx
{
    namespace reduce
    {
        template <typename _type_in>
        class cuda_reduce2D_1way_configs;


        typedef struct cu_reduce2D_1way_param_pack RWPK_2D;
    }
}


template <typename _Ty>
struct decx::reduce::cu_reduce1D_param_pack
{
    const void* _src;
    void* _dst;
    
    uint64_t _grid_len, _block_len;

    uint64_t _proc_len_v;
    uint64_t _proc_len_v1;


    cu_reduce1D_param_pack() {}

    cu_reduce1D_param_pack(const void*      src_ptr, 
                           void*            dst_ptr, 
                           const uint64_t   grid_len,
                           const uint64_t   block_len,
                           const uint64_t   proc_len_v, 
                           const uint64_t   proc_len_v1) :
        _src            (src_ptr),
        _dst            (dst_ptr),
        _grid_len       (grid_len),
        _block_len      (block_len),
        _proc_len_v     (proc_len_v),
        _proc_len_v1    (proc_len_v1)
    {}
};




struct decx::reduce::cu_reduce2D_1way_param_pack
{
    dim3 _grid_dims, _block_dims;

    const void* _src;
    void* _dst;

    // Could be in element or in scale of vector (e.g. v4, v8, etc.)
    uint32_t _calc_pitch_src;
    uint32_t _calc_pitch_dst;

    // Could be in element or in scale of vector (e.g. v4, v8, etc.)
    uint2 _calc_proc_dims;

    cu_reduce2D_1way_param_pack() {}


    cu_reduce2D_1way_param_pack(const dim3 grid_dims,               const dim3 block_dims,
                                const void* src_ptr,                void* dst_ptr, 
                                const uint32_t calc_pitch_src,      const uint32_t calc_pitch_dst, 
                                const uint2 calc_proc_dims) :

        _grid_dims          (grid_dims),
        _block_dims         (block_dims),
        _src                (src_ptr),
        _dst                (dst_ptr),
        _calc_pitch_src     (calc_pitch_src),
        _calc_pitch_dst     (calc_pitch_dst),
        _calc_proc_dims     (calc_proc_dims)
    {}

};




template <typename _type_in>
class decx::reduce::cuda_reduce1D_configs
{
private:
    decx::PtrInfo<void> _d_tmp1, _d_tmp2;

    _type_in _fill_val;

    uint64_t _actual_len;

    decx::alloc::MIF<void> _MIF_tmp1, _MIF_tmp2;

    std::vector<decx::reduce::RWPK_1D<_type_in>> _rwpks;

    template <bool _src_from_device>
    void _calc_kernel_param_packs();

    decx::reduce::RWPK_2D _rwpk_flatten;

    /**
    * The very beginning pointer of the whole process. If the source is not from device,
    * it will be set to this->_d_tmp1.ptr. Otherwise, it will be set to the device data array.
    */
    void* _proc_src;
    /**
    * The very end pointer of the whole process. If the source is not from device,
    * it will be set to this->get_leading_MIF().mem. Otherwise, it will be set to the device data array.
    */
    void* _proc_dst;

    bool _remain_load_byte;

    void inverse_mutex_MIF_states();

    // to private
    decx::alloc::MIF<void> get_leading_MIF() const;
    decx::alloc::MIF<void> get_lagging_MIF() const;

public:
    cuda_reduce1D_configs();

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


    uint64_t get_actual_len() const;


    void set_fill_val(const _type_in _val);

    
    _type_in get_fill_val() const;


    void* get_src();
    const void* get_dst();

    std::vector<decx::reduce::RWPK_1D<_type_in>>& get_rwpk();
    decx::reduce::RWPK_2D& get_rwpk_flatten();


    void release_buffer();


    void set_fp16_accuracy(const uint32_t _accu_lv);


    void set_cmp_or_not(const bool _is_cmp);


    ~cuda_reduce1D_configs();
};


namespace decx
{
    namespace reduce
    {
        void cuda_reduce1D_sum_fp32_caller_Async(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, decx::cuda_stream* S);
        void cuda_reduce1D_sum_fp64_caller_Async(decx::reduce::cuda_reduce1D_configs<double>* _kp_configs, decx::cuda_stream* S);
        void cuda_reduce1D_sum_i32_caller_Async(decx::reduce::cuda_reduce1D_configs<int32_t>* _kp_configs, decx::cuda_stream* S);
        void cuda_reduce1D_sum_u8_i32_caller_Async(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S);
        void cuda_reduce1D_sum_fp16_caller_Async(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S,
            const uint32_t _fp16_accu);



        template <bool _is_max>
        void cuda_reduce1D_cmp_fp32_caller_Async(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, decx::cuda_stream* S);
        template <bool _is_max>
        void cuda_reduce1D_cmp_int32_caller_Async(decx::reduce::cuda_reduce1D_configs<int32_t>* _kp_configs, decx::cuda_stream* S);
        template <bool _is_max>
        void cuda_reduce1D_cmp_fp64_caller_Async(decx::reduce::cuda_reduce1D_configs<double>* _kp_configs, decx::cuda_stream* S);
        template <bool _is_max>
        void cuda_reduce1D_cmp_fp16_caller_Async(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S);
        template <bool _is_max>
        void cuda_reduce1D_cmp_u8_caller_Async(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S);
    }
}




template <typename _type_in>
class decx::reduce::cuda_reduce2D_1way_configs
{
private:
    decx::Ptr2D_Info<void> _d_tmp1, _d_tmp2;

    // Only used when the source matrix is located on device
    uint32_t _Wdsrc;

    uint2 _proc_dims_actual;
    

    uint32_t _kernel_call_times;

    decx::alloc::MIF<void> _MIF_tmp1, _MIF_tmp2;

    std::vector<decx::reduce::RWPK_2D> _rwpks;

    template <bool _src_from_device>
    void _calc_kernel_h_param_packs(const bool _is_cmp = false);

    template <bool _src_from_device>
    void _calc_kernel_v_param_packs(const bool _is_cmp = false);

    bool _remain_load_byte;

    decx::Ptr2D_Info<void> _proc_src;
    void* _proc_dst;

    // to private
    decx::Ptr2D_Info<void> get_dtmp1() const;
    // to private
    decx::Ptr2D_Info<void> get_dtmp2() const;

    // to private
    uint2 get_actual_proc_dims() const;

    // to private
    void* get_leading_ptr() const;
    void* get_lagging_ptr() const;

    // to private
    void reverse_MIF_states();

public:
    cuda_reduce2D_1way_configs() {}

    
    template <bool _is_reduce_h>
    /**
    * @param proc_dims : The actual dimensions (measured in element) of the process area
    * @param S : The pointer of CUDA stream
    * @param _remain_load_byte : Does each kernel load the same data type.
    */
    void generate_configs(const uint2 proc_dims, decx::cuda_stream* S, const bool _remain_load_byte = false);


    template <bool _is_reduce_h>
    void generate_configs(decx::PtrInfo<void> dev_src, void* dst_ptr, const uint32_t Wdsrc, const uint2 proc_dims, 
        decx::cuda_stream* S, const bool _remain_load_byte = false);



    decx::Ptr2D_Info<void> get_src() const;
    void* get_dst() const;


    const std::vector<decx::reduce::cu_reduce2D_1way_param_pack>& get_rwpks() const;


    void set_cmp_or_not(const bool _is_cmp);
    

    void set_fp16_accuracy(const uint32_t _fp16_accu);


    void release_buffer();


    template <bool _is>
    void test();
};


namespace decx
{
    namespace reduce
    {
        void reduce_sum2D_h_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);
        void reduce_sum2D_h_fp64_Async(decx::reduce::cuda_reduce2D_1way_configs<double>* _configs, decx::cuda_stream* S);
        void reduce_sum2D_h_fp16_Async(decx::reduce::cuda_reduce2D_1way_configs<de::Half>* _configs, decx::cuda_stream* S, const uint32_t _fp16_accu);
        void reduce_sum2D_h_u8_i32_Async(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>* _configs, decx::cuda_stream* S);


        void reduce_sum2D_v_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);
        void reduce_sum2D_v_fp64_Async(decx::reduce::cuda_reduce2D_1way_configs<double>* _configs, decx::cuda_stream* S);
        void reduce_sum2D_v_fp16_Async(decx::reduce::cuda_reduce2D_1way_configs<de::Half>* _configs, decx::cuda_stream* S, const uint32_t _fp16_accu);
        void reduce_sum2D_v_u8_i32_Async(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>* _configs, decx::cuda_stream* S);

        // ------------------------------------------------- full -------------------------------------------
        /**
        * @return The pointer where the final value is stored
        */
        const void* reduce_sum2D_full_fp32_Async(decx::reduce::cuda_reduce1D_configs<float>* _configs, const void* src_ptr, const uint2 proc_dims,
            decx::cuda_stream* S, const bool _more_than_flatten);

        /**
        * @return The pointer where the final value is stored
        */
        const void* reduce_sum2D_full_i32_Async(decx::reduce::cuda_reduce1D_configs<int32_t>* _configs, const void* src_ptr, const uint2 proc_dims,
            decx::cuda_stream* S, const bool _more_than_flatten);

        /**
        * @return The pointer where the final value is stored
        */
        const void* reduce_sum2D_full_fp64_Async(decx::reduce::cuda_reduce1D_configs<double>* _configs, const void* src_ptr, const uint2 proc_dims,
            decx::cuda_stream* S, const bool _more_than_flatten);

        /**
        * @return The pointer where the final value is stored
        */
        const void* reduce_sum2D_full_fp16_fp32_Async(decx::reduce::cuda_reduce1D_configs<float>* _configs, const void* src_ptr, const uint2 proc_dims,
            decx::cuda_stream* S, const bool _more_than_flatten);

        /**
        * @return The pointer where the final value is stored
        */
        const void* reduce_sum2D_full_fp16_Async(decx::reduce::cuda_reduce1D_configs<de::Half>* _configs, const void* src_ptr, const uint2 proc_dims,
            decx::cuda_stream* S, const bool _more_than_flatten, const uint32_t _fp16_accu);


        /**
        * @return The pointer where the final value is stored
        */
        const void* reduce_sum2D_full_u8_i32_Async(decx::reduce::cuda_reduce1D_configs<int32_t>* _configs, const void* src_ptr, const uint2 proc_dims,
            decx::cuda_stream* S, const bool _more_than_flatten);


        template <bool _is_max>
        void reduce_cmp2D_h_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);
        template <bool _is_max>
        void reduce_cmp2D_h_fp16_Async(decx::reduce::cuda_reduce2D_1way_configs<de::Half>* _configs, decx::cuda_stream* S);
        template <bool _is_max>
        void reduce_cmp2D_h_u8_Async(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>* _configs, decx::cuda_stream* S);


        template <bool _is_max>
        void reduce_cmp2D_v_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);
        template <bool _is_max>
        void reduce_cmp2D_v_fp16_Async(decx::reduce::cuda_reduce2D_1way_configs<de::Half>* _configs, decx::cuda_stream* S);
        template <bool _is_max>
        void reduce_cmp2D_v_u8_Async(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>* _configs, decx::cuda_stream* S);

        // ---------------------------------------------- full --------------------------------------------

        /**
        * @return The pointer where the final value is stored
        */
        template <bool _is_max>
        const void* reduce_cmp2D_full_fp32_Async(decx::reduce::cuda_reduce1D_configs<float>* _configs, const void* src_ptr, const uint2 proc_dims,
            decx::cuda_stream* S, const bool _more_than_flatten);

        /**
        * @return The pointer where the final value is stored
        */
        template <bool _is_max>
        const void* reduce_cmp2D_full_int32_Async(decx::reduce::cuda_reduce1D_configs<int32_t>* _configs, const void* src_ptr, const uint2 proc_dims,
            decx::cuda_stream* S, const bool _more_than_flatten);

        /**
        * @return The pointer where the final value is stored
        */
        template <bool _is_max>
        const void* reduce_cmp2D_full_fp16_Async(decx::reduce::cuda_reduce1D_configs<de::Half>* _configs, const void* src_ptr, const uint2 proc_dims,
            decx::cuda_stream* S, const bool _more_than_flatten);

        /**
        * @return The pointer where the final value is stored
        */
        template <bool _is_max>
        const void* reduce_cmp2D_full_fp64_Async(decx::reduce::cuda_reduce1D_configs<double>* _configs, const void* src_ptr, const uint2 proc_dims,
            decx::cuda_stream* S, const bool _more_than_flatten);

        /**
        * @return The pointer where the final value is stored
        */
        template <bool _is_max>
        const void* reduce_cmp2D_full_u8_Async(decx::reduce::cuda_reduce1D_configs<uint8_t>* _configs, const void* src_ptr, const uint2 proc_dims,
            decx::cuda_stream* S, const bool _more_than_flatten);
    }
}


#endif