/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _BORDER_REFLECT_EXEC_PARAMS_
#define _BORDER_REFLECT_EXEC_PARAMS_


#include"../../../../core/basic.h"


namespace decx
{
    namespace bp {
        struct extend_reflect_exec_params
        {
            uint32_t _left, _right;
            uint32_t _actual_load_num_L;
            uint32_t _rightmost_0num_src;
            uint32_t _actual_load_num_R;
            uint32_t _L_v8_reflectL;
            uint32_t _L_v8_L;
        };


        void e_rfct_exep_gen_b32(decx::bp::extend_reflect_exec_params* _src, const uint32_t _left,
            const uint32_t _right, const size_t _actual_w_v1, const size_t _Wsrc_v8);


        void e_rfct_exep_gen_b8(decx::bp::extend_reflect_exec_params* _src, const uint32_t _left,
            const uint32_t _right, const size_t _actual_w_v1, const size_t _Wsrc_v8);


        void e_rfct_exep_gen_b16(decx::bp::extend_reflect_exec_params* _src, const uint32_t _left,
            const uint32_t _right, const size_t _actual_w_v1, const size_t _Wsrc_v8);


        void e_rfct_exep_gen_b64(decx::bp::extend_reflect_exec_params* _src, const uint32_t _left,
            const uint32_t _right, const size_t _actual_w_v1, const size_t _Wsrc_v8);


        uint32_t e_rfct_exep_get_buffer_len(const decx::bp::extend_reflect_exec_params* _src);


        __m256i e_rfct_exep_get_shufflevar_f_b32(const decx::bp::extend_reflect_exec_params* _src);


        __m256i e_rfct_exep_get_shufflevar_b_b32(const decx::bp::extend_reflect_exec_params* _src);


        __m256i e_rfct_exep_get_blend_b32(const decx::bp::extend_reflect_exec_params* _src);


        __m256i e_rfct_exep_get_shufflevar_f_b64(const decx::bp::extend_reflect_exec_params* _src);


        __m256i e_rfct_exep_get_shufflevar_b_b64(const decx::bp::extend_reflect_exec_params* _src);


        __m256i e_rfct_exep_get_blend_b64(const decx::bp::extend_reflect_exec_params* _src);


        __m128i e_rfct_exep_get_shufflevar_f_b8(const decx::bp::extend_reflect_exec_params* _src);


        __m128i e_rfct_exep_get_shufflevar_f_b16(const decx::bp::extend_reflect_exec_params* _src);


        __m128i e_rfct_exep_get_shufflevar_b_b16(const decx::bp::extend_reflect_exec_params* _src);


        __m128i e_rfct_exep_get_shufflevar_b_b8(const decx::bp::extend_reflect_exec_params* _src);


        __m128i e_rfct_exep_get_blend_b8(const decx::bp::extend_reflect_exec_params* _src);


        __m128i e_rfct_exep_get_blend_b16(const decx::bp::extend_reflect_exec_params* _src);
    }
}




#endif