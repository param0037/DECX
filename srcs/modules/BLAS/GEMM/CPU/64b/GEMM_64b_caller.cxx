/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "GEMM_64b_caller.h"


decx::ResourceHandle decx::_cpu_GEMM_b64_mgr;


template <> 
template <bool _is_cpl>
void decx::cpu_GEMM_general_config<double>::Run(decx::_Matrix* _A, 
                                                decx::_Matrix* _B, 
                                                decx::_Matrix* _dst)
{
    if (this->_L) {
        // rearrange matrix B
        decx::arrange_MatB_fp64_caller((double*)_B->Mat.ptr, 
                                       (double*)this->_arranged_B.ptr, 
                                       _B->Pitch(), 
                                       this->_arranged_B_dims.x, 
                                       _B->Height(), true, &this->_t2D, &this->_f_mgr_sort_B);

        // execute the GEMM_fp32 kernels
        decx::GEMM_AB_fp64_caller_8x<_is_cpl>((double*)_A->Mat.ptr,         (double*)this->_arranged_B.ptr, 
                                              (double*)_dst->Mat.ptr,       (double*)this->_extra_dst.ptr, 
                                               _A->Pitch(),                 this->_arranged_B_dims.x, 
                                               _dst->Pitch(),               this->_pitch_extdst * 8, 
                                               make_uint2((_B->Pitch() + 4) / 8, _dst->Height()), _A->Pitch(), 
                                               &this->_t2D);

        // copy the data from extra_cache_dst to _dst->Mat.ptr
        double* start_dst = DECX_PTR_SHF_XY<double, double>((double*)_dst->Mat.ptr, 
            make_uint2(0, this->_pre_f_mgrW.frag_len * (this->_t2D.thread_w - 1) * 8), _dst->Pitch());

        decx::gemm::CPUK::GEMM_fp64_cpy_L8((double*)this->_extra_dst.ptr,           start_dst, 
                                           this->_pitch_extdst * 8,                 _dst->Pitch(), 
                                           make_uint2(this->_pitch_extdst * 2 - 1,  _dst->Height()));
    }
    else {
        // rearrange matrix B
        decx::arrange_MatB_fp64_caller((double*)_B->Mat.ptr,                (double*)this->_arranged_B.ptr, 
                                       _B->Pitch(),                         this->_arranged_B_dims.x, 
                                       _B->Height(), false, &this->_t2D,    &this->_f_mgr_sort_B);

        // execute the GEMM_fp32 kernels
        decx::GEMM_AB_fp64_caller_16x<_is_cpl>((double*)_A->Mat.ptr,        (double*)this->_arranged_B.ptr, 
                                               (double*)_dst->Mat.ptr,      _A->Pitch(), 
                                               this->_arranged_B_dims.x,    _dst->Pitch(),
                                               make_uint2(_B->Pitch() / 8,  _dst->Height()), 
                                               _A->Pitch(),                 &this->_t2D);
    }
}

template void decx::cpu_GEMM_general_config<double>::Run<true>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*);
template void decx::cpu_GEMM_general_config<double>::Run<false>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*);



template <> 
template <bool _is_cpl>
void decx::cpu_GEMM_general_config<double>::Run(decx::_Matrix* _A, 
                                                decx::_Matrix* _B,
                                                decx::_Matrix* _C,
                                                decx::_Matrix* _dst)
{
    if (this->_L) {
        // rearrange matrix B
        decx::arrange_MatB_fp64_caller((double*)_B->Mat.ptr,    (double*)this->_arranged_B.ptr, 
                                        _B->Pitch(),            this->_arranged_B_dims.x, 
                                        _B->Height(),           true, 
                                        &this->_t2D, &this->_f_mgr_sort_B);

        // execute the GEMM_fp64 kernels
        decx::GEMM_ABC_fp64_caller_8x<_is_cpl>((double*)_A->Mat.ptr,    (double*)this->_arranged_B.ptr, 
                                               (double*)_C->Mat.ptr,    (double*)_dst->Mat.ptr, 
                                               (double*)this->_extra_dst.ptr,
                                               _A->Pitch(),             this->_arranged_B_dims.x, 
                                               _C->Pitch(),             _dst->Pitch(),
                                               this->_pitch_extdst * 8, make_uint2((_B->Pitch() + 4) / 8, _dst->Height()), 
                                               _A->Pitch(),             &this->_t2D);

        // copy the data from extra_cache_dst to _dst->Mat.ptr
        double* start_dst = DECX_PTR_SHF_XY<double, double>((double*)_dst->Mat.ptr, 
            make_uint2(0, this->_pre_f_mgrW.frag_len * (this->_t2D.thread_w - 1) * 8), _dst->Pitch());

        decx::gemm::CPUK::GEMM_fp64_cpy_L8((double*)this->_extra_dst.ptr,           start_dst, 
                                           this->_pitch_extdst * 8,                 _dst->Pitch(), 
                                           make_uint2(this->_pitch_extdst * 2 - 1,  _dst->Height()));
    }
    else {
        // rearrange matrix B
        decx::arrange_MatB_fp64_caller((double*)_B->Mat.ptr,    (double*)this->_arranged_B.ptr, 
                                        _B->Pitch(),            this->_arranged_B_dims.x, 
                                        _B->Height(),           false, &this->_t2D,     &this->_f_mgr_sort_B);

        // execute the GEMM_fp64 kernels
        decx::GEMM_ABC_fp64_caller_16x<_is_cpl>((double*)_A->Mat.ptr,       (double*)this->_arranged_B.ptr, 
                                                (double*)_C->Mat.ptr,       (double*)_dst->Mat.ptr,
                                                _A->Pitch(),                this->_arranged_B_dims.x, 
                                                _C->Pitch(),                _dst->Pitch(),
                                                make_uint2(_B->Pitch() / 8, _dst->Height()), 
                                                _A->Pitch(),                &this->_t2D);
    }
}

template void decx::cpu_GEMM_general_config<double>::Run<true>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, decx::_Matrix*);
template void decx::cpu_GEMM_general_config<double>::Run<false>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, decx::_Matrix*);


template <bool _is_cpl>
void decx::cpu::GEMM_64b(decx::_Matrix* _A, decx::_Matrix* _B, decx::_Matrix* _dst, de::DH* handle)
{
    if (decx::_cpu_GEMM_b64_mgr._res_ptr == NULL) {
        decx::_cpu_GEMM_b64_mgr.RegisterResource(new decx::cpu_GEMM_general_config<double>,
            5, &decx::cpu_GEMM_general_config<float>::release_buffers);
    }
    decx::_cpu_GEMM_b64_mgr.lock();

    decx::cpu_GEMM_general_config<double>* _planner =
        decx::_cpu_GEMM_b64_mgr.get_resource_raw_ptr<decx::cpu_GEMM_general_config<double>>();

    if (_planner->Changed(decx::cpu::_get_permitted_concurrency(), &_A->get_layout(), &_B->get_layout()))
    {
        // Check the dims and types of the input matrices
        decx::cpu_GEMM_general_config<double>::Validate(handle, &_A->get_layout(), &_B->get_layout());
        if (handle->error_type != decx::DECX_error_types::DECX_SUCCESS) {
            return;
        }
        // Configure the tunning parameters
        _planner->Config(decx::cpu::_get_permitted_concurrency(), &_A->get_layout(), &_B->get_layout(), handle);
    }
    // Run the GEMM proceudre
    _planner->Run<0>(_A, _B, _dst);

    decx::_cpu_GEMM_b64_mgr.unlock();
}

template void decx::cpu::GEMM_64b<true>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, de::DH*);
template void decx::cpu::GEMM_64b<false>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, de::DH*);


template <bool _is_cpl>
void decx::cpu::GEMM_64b_ABC(decx::_Matrix* _A, decx::_Matrix* _B, decx::_Matrix* _C, decx::_Matrix* _dst, de::DH* handle)
{
    if (decx::_cpu_GEMM_b64_mgr._res_ptr == NULL) {
        decx::_cpu_GEMM_b64_mgr.RegisterResource(new decx::cpu_GEMM_general_config<double>,
            5, &decx::cpu_GEMM_general_config<double>::release_buffers);
    }
    decx::_cpu_GEMM_b64_mgr.lock();

    decx::cpu_GEMM_general_config<double>* _planner =
        decx::_cpu_GEMM_b64_mgr.get_resource_raw_ptr<decx::cpu_GEMM_general_config<double>>();

    if (_planner->Changed(decx::cpu::_get_permitted_concurrency(), &_A->get_layout(), &_B->get_layout()))
    {
        // Check the dims and types of the input matrices
        decx::cpu_GEMM_general_config<double>::Validate(handle, &_A->get_layout(), &_B->get_layout(), &_C->get_layout());
        if (handle->error_type != decx::DECX_error_types::DECX_SUCCESS) {
            return;
        }
        // Configure the tunning parameters
        _planner->Config(decx::cpu::_get_permitted_concurrency(), &_A->get_layout(), &_B->get_layout(), handle);
    }
    // Run the GEMM proceudre
    _planner->Run<0>(_A, _B, _dst);

    decx::_cpu_GEMM_b64_mgr.unlock();
}

template void decx::cpu::GEMM_64b_ABC<true>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, de::DH*);
template void decx::cpu::GEMM_64b_ABC<false>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, de::DH*);
