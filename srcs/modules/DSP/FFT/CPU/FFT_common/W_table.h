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


#ifndef _W_TABLE_H_
#define _W_TABLE_H_


#include "../../FFT_commons.h"
#include "../../../../../common/Classes/classes_util.h"



namespace decx
{
namespace dsp
{
namespace fft
{
    _THREAD_CALL_ static uint64_t _calc_WT_mapping(const uint64_t warp_loc_id,
        const uint64_t _signal_len, const uint64_t warp_proc_len)
    {
#define _CALC_MAPPING_(_idx) ((_idx) > warp_proc_len ? ((_idx) - warp_proc_len) : (_idx))
        return (_signal_len / warp_proc_len) * _CALC_MAPPING_(warp_loc_id);
#ifdef _CALC_MAPPING_
#undef _CALC_MAPPING_
#endif
    }

    _THREAD_CALL_  static __m128i _calc_WT_mapping_v4(const __m128i _base_dex, const uint64_t warp_loc_id,
        const uint64_t _signal_len, const uint64_t _warp_proc_len)
    {
        const __m128i _warp_proc_len_v4 = _mm_set1_epi32(_warp_proc_len);

        __m128i _crit = _mm_cmpgt_epi32(_base_dex, _warp_proc_len_v4);

        __m128i _mid = _mm_mullo_epi32(_mm_sub_epi32(_base_dex, _mm_and_si128(_warp_proc_len_v4, _crit)),
            _mm_set1_epi32(_signal_len / _warp_proc_len));

        _crit = _mm_cmplt_epi32(_base_dex, _mm_set1_epi32(_signal_len));
        return _mm_and_si128(_mid, _crit);
    }

    namespace CPUK {
        _THREAD_FUNCTION_ void _W_table_gen_cplxf(double* __restrict _W_table, const uint32_t _proc_len_v4,
            const uint64_t _signal_length, const uint64_t _index_start);


        _THREAD_FUNCTION_ void _W_table_gen_cplxd(de::CPd* __restrict _W_table, const uint32_t _proc_len_v2,
            const uint64_t _signal_length, const uint64_t _index_start);
    }


    template <typename _cplx_type>
    class Rotational_Factors_Table;
}
}
}


template <typename _cplx_type>
class decx::dsp::fft::Rotational_Factors_Table
{
private:
    uint64_t _actual_len;
    uint64_t _alloc_len;

    decx::PtrInfo<void> _W_table;

    void _realloc_table(const uint64_t _new_len, de::DH* handle);


    void _alloc_table_from_scratch(const uint64_t _new_len, de::DH* handle);

public:

    Rotational_Factors_Table() {}
    

    void _alloc_table(const uint64_t _len, de::DH *handle);


    void _generate_table(decx::utils::_thr_1D* t1D);


    void _release();


    template <typename _ptr_type>
    const _ptr_type* _get_table_ptr() const
    {
        return (const _ptr_type*)this->_W_table.ptr;
    }


    ~Rotational_Factors_Table();
};



#endif