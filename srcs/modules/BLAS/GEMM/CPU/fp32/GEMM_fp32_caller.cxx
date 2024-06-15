/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "GEMM_fp32_caller.h"


decx::ResourceHandle decx::_cpu_GEMM_fp32_mgr;


template <> template <>
void decx::cpu_GEMM_general_config<float>::Run<0>(decx::_Matrix* _A, 
                                               decx::_Matrix* _B, 
                                               decx::_Matrix* _dst)
{
    if (this->_L) {
        // rearrange matrix B
        decx::gemm::CPUK::arrange_MatB_fp32_caller((float*)_B->Mat.ptr,         
                                                    (float*)this->_arranged_B.ptr, 
                                                    _B->Pitch(), 
                                                    this->_arranged_B_dims.x, 
                                                    _B->Height(), 
                                                    true, &this->_t2D, &this->_f_mgr_sort_B);

        // execute the GEMM_fp32 kernels
        decx::GEMM_AB_fp32_caller_8x((float*)_A->Mat.ptr,                                   (float*)this->_arranged_B.ptr, 
                                     (float*)_dst->Mat.ptr,                                 (float*)this->_extra_dst.ptr, 
                                     _A->Pitch(),                                           this->_arranged_B_dims.x, 
                                     _dst->Pitch(),                                         this->_pitch_extdst * 16, 
                                     make_uint2((_B->Pitch() + 8) / 16, _dst->Height()),    _A->Pitch(), 
                                     &this->_t2D);
        
        // copy the data from extra_cache_dst to _dst->Mat.ptr
        float* start_dst = DECX_PTR_SHF_XY<float, float>((float*)_dst->Mat.ptr, 
            make_uint2(0, this->_pre_f_mgrW.frag_len * (this->_t2D.thread_w - 1) * 16), _dst->Pitch());

        decx::gemm::CPUK::GEMM_fp32_cpy_L8((float*)this->_extra_dst.ptr,        start_dst, 
                                           this->_pitch_extdst * 16,    _dst->Pitch(), 
                                           make_uint2(this->_pitch_extdst * 2 - 1, _dst->Height()));
    }
    else {
        // rearrange matrix B
        decx::gemm::CPUK::arrange_MatB_fp32_caller((float*)_B->Mat.ptr,         (float*)this->_arranged_B.ptr, 
                                                    _B->Pitch(),                this->_arranged_B_dims.x, 
                                                    _B->Height(),               false, 
                                                    &this->_t2D,                &this->_f_mgr_sort_B);

        // execute the GEMM_fp32 kernels
        decx::GEMM_AB_fp32_caller_16x((float*)_A->Mat.ptr,              (float*)this->_arranged_B.ptr, 
                                      (float*)_dst->Mat.ptr,            _A->Pitch(), 
                                      this->_arranged_B_dims.x,         _dst->Pitch(),
                                      make_uint2(_B->Pitch() / 16, _dst->Height()), 
                                      _A->Pitch(), &this->_t2D);
    }
}


template <> template <>
void decx::cpu_GEMM_general_config<float>::Run<0>(decx::_Matrix* _A, 
                                               decx::_Matrix* _B,
                                               decx::_Matrix* _C,
                                               decx::_Matrix* _dst)
{
    if (this->_L) {
        // rearrange matrix B
        decx::gemm::CPUK::arrange_MatB_fp32_caller((float*)_B->Mat.ptr,         
                                                    (float*)this->_arranged_B.ptr, 
                                                    _B->Pitch(), 
                                                    this->_arranged_B_dims.x, 
                                                    _B->Height(), 
                                                    true, &this->_t2D, &this->_f_mgr_sort_B);

        // execute the GEMM_fp32 kernels
        decx::GEMM_ABC_fp32_caller_8x((float*)_A->Mat.ptr,          (float*)this->_arranged_B.ptr, 
                                      (float*)_C->Mat.ptr,          (float*)_dst->Mat.ptr, 
                                      (float*)this->_extra_dst.ptr,         _A->Pitch(), 
                                      this->_arranged_B_dims.x,     _C->Pitch(), 
                                      _dst->Pitch(),                this->_pitch_extdst * 16, 
                                      make_uint2((_B->Pitch() + 8) / 16, _dst->Height()), 
                                      _A->Pitch(), &this->_t2D);
        
        // copy the data from extra_cache_dst to _dst->Mat.ptr
        float* start_dst = DECX_PTR_SHF_XY<float, float>((float*)_dst->Mat.ptr, 
            make_uint2(0, this->_pre_f_mgrW.frag_len * (this->_t2D.thread_w - 1) * 16), _dst->Pitch());

        decx::gemm::CPUK::GEMM_fp32_cpy_L8((float*)this->_extra_dst.ptr,        start_dst, 
                                           this->_pitch_extdst * 16,    _dst->Pitch(), 
                                           make_uint2(this->_pitch_extdst * 2 - 1, _dst->Height()));
    }
    else {
        // generate the configuration for sorting matrix B
        decx::utils::frag_manager_gen(&this->_f_mgr_sort_B, this->_arranged_B_dims.y, this->_t2D.total_thread);

        // rearrange matrix B
        decx::gemm::CPUK::arrange_MatB_fp32_caller((float*)_B->Mat.ptr,         (float*)this->_arranged_B.ptr, 
                                                    _B->Pitch(),                this->_arranged_B_dims.x, 
                                                    _B->Height(),               false, 
                                                    &this->_t2D,                &this->_f_mgr_sort_B);

        // execute the GEMM_fp32 kernels
        decx::GEMM_ABC_fp32_caller_16x((float*)_A->Mat.ptr,         (float*)this->_arranged_B.ptr, 
                                       (float*)_C->Mat.ptr,         (float*)_dst->Mat.ptr,
                                       _A->Pitch(),                 this->_arranged_B_dims.x, 
                                       _C->Pitch(),                 _dst->Pitch(),
                                       make_uint2(_B->Pitch() / 16, _dst->Height()), 
                                       _A->Pitch(),                 &this->_t2D);
    }
}


void
decx::cpu::GEMM_fp32(decx::_Matrix* _A, decx::_Matrix* _B, decx::_Matrix* _dst, de::DH *handle)
{
    if (decx::_cpu_GEMM_fp32_mgr._res_ptr == NULL) {
        decx::_cpu_GEMM_fp32_mgr.RegisterResource(new decx::cpu_GEMM_general_config<float>,
            5, &decx::cpu_GEMM_general_config<float>::release_buffers);
    }
    decx::_cpu_GEMM_fp32_mgr.lock();

    decx::cpu_GEMM_general_config<float>* _planner =
        decx::_cpu_GEMM_fp32_mgr.get_resource_raw_ptr<decx::cpu_GEMM_general_config<float>>();
    
    if (_planner->Changed(decx::cpu::_get_permitted_concurrency(), &_A->get_layout(), &_B->get_layout())) 
    {
        // Check the dims and types of the input matrices
        decx::cpu_GEMM_general_config<float>::Validate(handle, &_A->get_layout(), &_B->get_layout());
        if (handle->error_type != decx::DECX_error_types::DECX_SUCCESS) {
            return;
        }
        // Configure the tunning parameters
        _planner->Config(decx::cpu::_get_permitted_concurrency(), &_A->get_layout(), &_B->get_layout(), handle);
    }
    // Run the GEMM proceudre
    _planner->Run<0>(_A, _B, _dst);

    decx::_cpu_GEMM_fp32_mgr.unlock();
}


void
decx::cpu::GEMM_fp32_ABC(decx::_Matrix* _A, decx::_Matrix* _B, decx::_Matrix* _C, decx::_Matrix* _dst, de::DH *handle)
{
    if (decx::_cpu_GEMM_fp32_mgr._res_ptr == NULL) {
        decx::_cpu_GEMM_fp32_mgr.RegisterResource(new decx::cpu_GEMM_general_config<float>,
            5, &decx::cpu_GEMM_general_config<float>::release_buffers);
    }
    decx::_cpu_GEMM_fp32_mgr.lock();

    decx::cpu_GEMM_general_config<float>* _planner =
        decx::_cpu_GEMM_fp32_mgr.get_resource_raw_ptr<decx::cpu_GEMM_general_config<float>>();

    if (_planner->Changed(decx::cpu::_get_permitted_concurrency(), &_A->get_layout(), &_B->get_layout()))
    {
        // Check the dims and types of the input matrices
        decx::cpu_GEMM_general_config<float>::Validate(handle, &_A->get_layout(), &_B->get_layout(), &_C->get_layout());
        if (handle->error_type != decx::DECX_error_types::DECX_SUCCESS) {
            return;
        }
        // Configure the tunning parameters
        _planner->Config(decx::cpu::_get_permitted_concurrency(), &_A->get_layout(), &_B->get_layout(), handle);
    }
    // Run the GEMM proceudre
    _planner->Run<0>(_A, _B, _C, _dst);

    decx::_cpu_GEMM_fp32_mgr.unlock();
}
