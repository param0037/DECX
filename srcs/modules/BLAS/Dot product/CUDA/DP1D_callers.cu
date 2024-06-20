/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "DP1D_callers.cuh"


const void* decx::blas::cuda_DP1D_fp32_caller_Async(decx::blas::cuda_DP1D_configs<float>* _configs, decx::cuda_stream* S)
{
    const void* res_ptr = _configs->postproc_needed() ? _configs->get_configs_ptr<float>()->get_src() : _configs->_dev_dst.ptr;

    decx::blas::GPUK::cu_block_dot1D_fp32 << <_configs->get_grid_len_k1(), _REDUCE1D_BLOCK_DIM_, 0, S->get_raw_stream_ref() >> > (
        (float4*)_configs->_dev_A.ptr, (float4*)_configs->_dev_B.ptr, (float*)res_ptr,
        decx::utils::ceil<uint64_t>(_configs->get_actual_proc_len(), _CU_REDUCE1D_MEM_ALIGN_4B_), _configs->get_actual_proc_len());

    if (_configs->postproc_needed()) {
        decx::reduce::cuda_reduce1D_configs<float>* _postproc_configs = _configs->get_configs_ptr<float>();

        std::vector<decx::reduce::RWPK_1D<float>>& _rwpk_arr = _postproc_configs->get_rwpk();
        decx::reduce::RWPK_1D<float> _rwpk;

        for (int i = 0; i < _rwpk_arr.size(); ++i) {
            _rwpk = _rwpk_arr[i];

            decx::reduce::GPUK::cu_block_reduce_sum1D_fp32 << <_rwpk._grid_len, _rwpk._block_len,
                0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (float*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
        }

        return _postproc_configs->get_dst();
    }
    else {
        return _configs->_dev_dst.ptr;
    }
}


const void* decx::blas::cuda_DP1D_fp16_caller_Async(decx::blas::cuda_DP1D_configs<de::Half>* _configs, 
    decx::cuda_stream* S, const uint32_t _fp16_accu)
{
    const void* res_ptr = _configs->postproc_needed() ? _configs->get_configs_ptr<de::Half>()->get_src() : _configs->_dev_dst.ptr;

    switch (_fp16_accu)
    {
    case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1:
        decx::blas::GPUK::cu_block_dot1D_fp16_L1 << <_configs->get_grid_len_k1(), _REDUCE1D_BLOCK_DIM_, 0, S->get_raw_stream_ref() >> > (
            (float4*)_configs->_dev_A.ptr, (float4*)_configs->_dev_B.ptr, (float*)res_ptr,
            decx::utils::ceil<uint64_t>(_configs->get_actual_proc_len(), _CU_REDUCE1D_MEM_ALIGN_2B_), _configs->get_actual_proc_len());
        break;

    case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L2:
        decx::blas::GPUK::cu_block_dot1D_fp16_L2 << <_configs->get_grid_len_k1(), _REDUCE1D_BLOCK_DIM_, 0, S->get_raw_stream_ref() >> > (
            (float4*)_configs->_dev_A.ptr, (float4*)_configs->_dev_B.ptr, (__half*)res_ptr,
            decx::utils::ceil<uint64_t>(_configs->get_actual_proc_len(), _CU_REDUCE1D_MEM_ALIGN_2B_), _configs->get_actual_proc_len());
        break;

    case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L3:
        decx::blas::GPUK::cu_block_dot1D_fp16_L3 << <_configs->get_grid_len_k1(), _REDUCE1D_BLOCK_DIM_, 0, S->get_raw_stream_ref() >> > (
            (float4*)_configs->_dev_A.ptr, (float4*)_configs->_dev_B.ptr, (__half*)res_ptr,
            decx::utils::ceil<uint64_t>(_configs->get_actual_proc_len(), _CU_REDUCE1D_MEM_ALIGN_2B_), _configs->get_actual_proc_len());
        break;
    default:
        break;
    }
    
    if (_configs->postproc_needed()) 
    {
        if (_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) {
            decx::reduce::cuda_reduce1D_configs<float>* _postproc_configs = _configs->get_configs_ptr<float>();

            std::vector<decx::reduce::RWPK_1D<float>>& _rwpk_arr = _postproc_configs->get_rwpk();
            decx::reduce::RWPK_1D<float> _rwpk;

            for (int i = 0; i < _rwpk_arr.size(); ++i) {
                _rwpk = _rwpk_arr[i];
                decx::reduce::GPUK::cu_block_reduce_sum1D_fp32 << <_rwpk._grid_len, _rwpk._block_len,
                    0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (float*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
            }
            return _postproc_configs->get_dst();
        }
        else{
            decx::reduce::cuda_reduce1D_configs<de::Half>* _postproc_configs = _configs->get_configs_ptr<de::Half>();

            std::vector<decx::reduce::RWPK_1D<de::Half>>& _rwpk_arr = _postproc_configs->get_rwpk();
            decx::reduce::RWPK_1D<de::Half> _rwpk;

            for (int i = 0; i < _rwpk_arr.size(); ++i) {
                _rwpk = _rwpk_arr[i];
                switch (_fp16_accu)
                {
                case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L2:
                    decx::reduce::GPUK::cu_block_reduce_sum1D_fp16_L2 << <_rwpk._grid_len, _rwpk._block_len,
                        0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (__half*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
                    break;

                case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L3:
                    decx::reduce::GPUK::cu_block_reduce_sum1D_fp16_L3 << <_rwpk._grid_len, _rwpk._block_len,
                        0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (__half*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
                    break;
                default:
                    break;
                }
            }
            return _postproc_configs->get_dst();
        }
    }
    else {
        return _configs->_dev_dst.ptr;
    }
}





const void* decx::blas::cuda_DP1D_fp64_caller_Async(decx::blas::cuda_DP1D_configs<double>* _configs, decx::cuda_stream* S)
{
    const void* res_ptr = _configs->postproc_needed() ? _configs->get_configs_ptr<double>()->get_src() : _configs->_dev_dst.ptr;

    decx::blas::GPUK::cu_block_dot1D_fp64 << <_configs->get_grid_len_k1(), _REDUCE1D_BLOCK_DIM_, 0, S->get_raw_stream_ref() >> > (
        (double2*)_configs->_dev_A.ptr, (double2*)_configs->_dev_B.ptr, (double*)res_ptr,
        decx::utils::ceil<uint64_t>(_configs->get_actual_proc_len(), _CU_REDUCE1D_MEM_ALIGN_8B_), _configs->get_actual_proc_len());

    if (_configs->postproc_needed()) {
        decx::reduce::cuda_reduce1D_configs<double>* _postproc_configs = _configs->get_configs_ptr<double>();

        std::vector<decx::reduce::RWPK_1D<double>>& _rwpk_arr = _postproc_configs->get_rwpk();
        decx::reduce::RWPK_1D<double> _rwpk;

        for (int i = 0; i < _rwpk_arr.size(); ++i) {
            _rwpk = _rwpk_arr[i];

            decx::reduce::GPUK::cu_block_reduce_sum1D_fp64 << <_rwpk._grid_len, _rwpk._block_len,
                0, S->get_raw_stream_ref() >> > ((double2*)_rwpk._src, (double*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
        }

        return _postproc_configs->get_dst();
    }
    else {
        return _configs->_dev_dst.ptr;
    }
}




const void* decx::blas::cuda_DP1D_cplxf_caller_Async(decx::blas::cuda_DP1D_configs<double>* _configs, decx::cuda_stream* S)
{
    const void* res_ptr = _configs->postproc_needed() ? _configs->get_configs_ptr<double>()->get_src() : _configs->_dev_dst.ptr;

    decx::blas::GPUK::cu_block_dot1D_cplxf << <_configs->get_grid_len_k1(), _REDUCE1D_BLOCK_DIM_, 0, S->get_raw_stream_ref() >> > (
        (float4*)_configs->_dev_A.ptr, (float4*)_configs->_dev_B.ptr, (de::CPf*)res_ptr,
        decx::utils::ceil<uint64_t>(_configs->get_actual_proc_len(), _CU_REDUCE1D_MEM_ALIGN_8B_), _configs->get_actual_proc_len());

    if (_configs->postproc_needed()) {
        decx::reduce::cuda_reduce1D_configs<double>* _postproc_configs = _configs->get_configs_ptr<double>();

        std::vector<decx::reduce::RWPK_1D<double>>& _rwpk_arr = _postproc_configs->get_rwpk();
        decx::reduce::RWPK_1D<double> _rwpk;

        for (int i = 0; i < _rwpk_arr.size(); ++i) {
            _rwpk = _rwpk_arr[i];

            decx::reduce::GPUK::cu_block_reduce_sum1D_cplxf << <_rwpk._grid_len, _rwpk._block_len,
                0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (de::CPf*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
        }

        return _postproc_configs->get_dst();
    }
    else {
        return _configs->_dev_dst.ptr;
    }
}