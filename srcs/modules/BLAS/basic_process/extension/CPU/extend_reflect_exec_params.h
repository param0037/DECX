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