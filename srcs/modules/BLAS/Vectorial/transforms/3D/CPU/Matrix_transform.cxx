/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Matrix_transform.h"



_THREAD_FUNCTION_ void
decx::tf::CPUK::_vec4_mul_mat4x4_fp32_1D(const float* __restrict        src, 
                                         float* __restrict              dst, 
                                         const decx::_Mat4x4f           _tf_mat,
                                         const size_t                   _proc_len)
{
    size_t loc_dex = 0;

    __m128 reg;
    decx::_Vector4f recv, store;
    store._vec = _mm_set1_ps(0);

    for (int i = 0; i < _proc_len; ++i) {
        recv._vec = _mm_load_ps(src + loc_dex);

        reg = _mm_permute_ps(recv._vec, _MM_SHUFFLE(0, 0, 0, 0));           // col0
        store._vec = _mm_fmadd_ps(reg, _tf_mat._col[0], store._vec);        // row0

        reg = _mm_permute_ps(recv._vec, _MM_SHUFFLE(1, 1, 1, 1));           // col1
        store._vec = _mm_fmadd_ps(reg, _tf_mat._col[1], store._vec);        // row1

        reg = _mm_permute_ps(recv._vec, _MM_SHUFFLE(2, 2, 2, 2));           // col2
        store._vec = _mm_fmadd_ps(reg, _tf_mat._col[2], store._vec);        // row2

        reg = _mm_permute_ps(recv._vec, _MM_SHUFFLE(3, 3, 3, 3));           // col3
        store._vec = _mm_fmadd_ps(reg, _tf_mat._col[3], store._vec);        // row3

        _mm_store_ps(dst + loc_dex, store._vec);
        loc_dex += 4;
    }
}




_THREAD_FUNCTION_ void
decx::tf::CPUK::_vec3_mul_mat4x3_fp32_1D(const float* __restrict        src, 
                                         float* __restrict              dst, 
                                         const decx::_Mat4x4f           _tf_mat,
                                         const size_t                   _proc_len)
{
    size_t loc_dex = 0;

    __m128 reg;
    decx::_Vector4f recv, store;
    store._vec = _mm_set1_ps(0);

    for (int i = 0; i < _proc_len; ++i) {
        recv._vec = _mm_load_ps(src + loc_dex);

        reg = _mm_permute_ps(recv._vec, _MM_SHUFFLE(0, 0, 0, 0));           // col0
        store._vec = _mm_fmadd_ps(reg, _tf_mat._col[0], store._vec);        // row0

        reg = _mm_permute_ps(recv._vec, _MM_SHUFFLE(1, 1, 1, 1));           // col1
        store._vec = _mm_fmadd_ps(reg, _tf_mat._col[1], store._vec);        // row1

        reg = _mm_permute_ps(recv._vec, _MM_SHUFFLE(2, 2, 2, 2));           // col2
        store._vec = _mm_fmadd_ps(reg, _tf_mat._col[2], store._vec);        // row2

        // set the last element of the destinated vector to zero anyway
        // don't need to be calculated
        store._vec = _mm_castsi128_ps(_mm_and_si128(
            _mm_castps_si128(store._vec), _mm_setr_epi32(0xffffffff, 0xffffffff, 0xffffffff, 0x0)));

        _mm_store_ps(dst + loc_dex, store._vec);
        loc_dex += 4;
    }
}



void decx::tf::Vec4_transform(decx::_Vector* src, decx::_Vector* dst, decx::_Matrix* transform_matrix, de::DH* handle)
{

}



_DECX_API_ de::DH 
de::tf::cpu::Vec_transform(de::Vector& src, de::Vector& dst, de::Matrix& transform_matrix)
{
    de::DH handle;
    
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Matrix* _transform_matrix = dynamic_cast<decx::_Matrix*>(&transform_matrix);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (_src->type == de::_DATA_TYPES_FLAGS_::_VECTOR4F_ || _src->type == de::_DATA_TYPES_FLAGS_::_VECTOR3F_) {
        const size_t _proc_len = _src->_length / 4;

        decx::_Mat4x4f _tf_mat4_by_4;
        for (int i = 0; i < 4; ++i) {
            _tf_mat4_by_4._row[i] = _mm_load_ps((float*)_transform_matrix->Mat.ptr + i * _transform_matrix->Pitch());
        }

        decx::mat::_mat4x4_transpose_fp32(&_tf_mat4_by_4);      // transpose the 4_by_4 matrix

        decx::tf::CPUK::_vec_mul_mat4_kernel1D_ptr kernel = NULL;
        if (_src->type == de::_DATA_TYPES_FLAGS_::_VECTOR4F_) {
            kernel = decx::tf::CPUK::_vec4_mul_mat4x4_fp32_1D;
        }
        else {
            kernel = decx::tf::CPUK::_vec3_mul_mat4x3_fp32_1D;
        }

        decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
        decx::utils::frag_manager f_mgr;
        decx::utils::frag_manager_gen(&f_mgr, _proc_len, t1D.total_thread);

        const float* _loc_src_ptr = (float*)_src->Vec.ptr;
        float* _loc_dst_ptr = (float*)_dst->Vec.ptr;
        for (int i = 0; i < t1D.total_thread - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task_default(kernel, _loc_src_ptr, _loc_dst_ptr,
                _tf_mat4_by_4, f_mgr.frag_len);

            _loc_src_ptr += f_mgr.frag_len * 4;
            _loc_dst_ptr += f_mgr.frag_len * 4;
        }
        size_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(kernel, _loc_src_ptr, _loc_dst_ptr,
            _tf_mat4_by_4, _L);
    }
    else {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
            TYPE_ERROR_NOT_MATCH);
        return handle;
    }

    decx::err::Success(&handle);
    return handle;
}