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


#ifndef _CPU_FFT1D_PLANNER_H_
#define _CPU_FFT1D_PLANNER_H_


#include "../../../../../common/basic.h"
#include "../FFT_common/CPU_FFT_tiles.h"
#include "../FFT_common/W_table.h"
#include "../../../../../common/Array/Fixed_Length_Array.h"
#include "../FFT_common/CPU_FFT_defs.h"
#include "../../../../../common/Classes/Vector.h"
#include "../../../../core/resources_manager/decx_resource.h"


namespace decx
{
namespace dsp
{
namespace fft 
{
    template <typename _type_in>
    class cpu_FFT1D_planner;


    template <typename _type_in>
    class cpu_FFT1D_smaller;
}
}
}


template <typename _type_in>
class decx::dsp::fft::cpu_FFT1D_smaller
{
private:
    uint32_t _signal_length;

    std::vector<decx::dsp::fft::FKI1D> _kernel_infos;

    std::vector<uint32_t> _radixes;

    decx::dsp::fft::Rotational_Factors_Table<_type_in> _W_table;

    decx::utils::frag_manager _thread_dispatch;

public:
    cpu_FFT1D_smaller() {}


    _CRSR_ cpu_FFT1D_smaller(const uint32_t signal_length, de::DH* handle);


    _CRSR_ void set_length(const uint32_t signal_length, de::DH* handle);



    _CRSR_ const decx::dsp::fft::FKI1D* get_kernel_info_ptr(const uint32_t _id) const;


    _CRSR_ void plan(decx::utils::_thr_1D* t1D);


    uint32_t get_signal_len() const;


    uint32_t get_kernel_call_num() const;


    const decx::utils::frag_manager* get_thread_patching() const;


    decx::utils::frag_manager* get_thread_patching_modify();


    template <typename _ptr_type>
    const _ptr_type* get_W_table() const
    {
        return this->_W_table.template _get_table_ptr<_ptr_type>();
    }


    ~cpu_FFT1D_smaller();
};



template <typename _data_type>
class decx::dsp::fft::cpu_FFT1D_planner
{
private:
    uint64_t _signal_length;

    decx::PtrInfo<void> _tmp1, _tmp2;
    decx::alloc::MIF<void> _MIF1, _MIF2;

    bool _without_larger_DFT;


    uint32_t _permitted_concurrency;


    std::vector<decx::dsp::fft::FKT1D> _tiles;


    std::vector<uint32_t> _all_radixes;

    
    decx::utils::Fixed_Length_Array<decx::dsp::fft::cpu_FFT1D_smaller<_data_type>> _smaller_FFTs;
    
    

    void _allocate_spaces(de::DH *handle);


    void _apart_for_smaller_FFTs(de::DH* handle);

public:
    std::vector<decx::dsp::fft::FKI1D> _outer_kernel_info;
    cpu_FFT1D_planner() {}

    
    bool changed(const uint64_t signal_len, const uint32_t concurrency) const;


    uint64_t get_signal_len() const;


    void set_signal_length(const uint64_t signal_length);


    template <typename _type_in>
    void Forward(decx::_Vector* src, decx::_Vector* dst, decx::utils::_thread_arrange_1D* t1D) const;


    template <typename _type_out>
    void Inverse(decx::_Vector* src, decx::_Vector* dst, decx::utils::_thread_arrange_1D* t1D) const;


    void plan(const uint64_t signal_len, decx::utils::_thr_1D* t1D, de::DH *handle);


    const decx::dsp::fft::FKT1D* get_tile_ptr(const uint32_t _thread_id) const;


    const decx::dsp::fft::cpu_FFT1D_smaller<_data_type>* get_smaller_FFT_info_ptr(const uint32_t _order) const;


    const decx::dsp::fft::FKI1D* get_outer_kernel_info(const uint32_t _order) const;


    uint32_t get_kernel_call_num() const;


    void* get_tmp1_ptr() const;
    void* get_tmp2_ptr() const;


    static void release_buffers(decx::dsp::fft::cpu_FFT1D_planner<_data_type>* _fake_this);


    ~cpu_FFT1D_planner();
};



namespace decx
{
    namespace dsp {
        namespace fft {
            typedef struct CPU_FFT1D_Intermediate_Twd FIMT1D;
        }
    }
}


struct decx::dsp::fft::CPU_FFT1D_Intermediate_Twd
{
    uint64_t _previous_fact_sum;
    uint32_t _next_FFT_len;

    uint64_t _signal_length;


    CPU_FFT1D_Intermediate_Twd(const uint64_t _signal_len, const uint32_t _first_FFT_len, const uint32_t _next_FFT_len = 1)
    {
        this->_signal_length = _signal_len;
        this->_previous_fact_sum = _first_FFT_len;
        this->_next_FFT_len = _next_FFT_len;
    }


    void update(const uint32_t _next_FFT_len) 
    {
        this->_previous_fact_sum *= this->_next_FFT_len;
        this->_next_FFT_len = _next_FFT_len;
    }


    uint64_t divide_length() const
    {
        return this->_previous_fact_sum * this->_next_FFT_len;
    }



    uint64_t gap() const
    {
        return this->_signal_length / this->_previous_fact_sum / this->_next_FFT_len;
    }
};


namespace decx
{
    namespace dsp {
        namespace fft {
            extern decx::ResourceHandle g_cpu_FFT1D_cplxf32_planner;
            extern decx::ResourceHandle g_cpu_IFFT1D_cplxf32_planner;

            extern decx::ResourceHandle g_cpu_FFT1D_cplxd64_planner;
            extern decx::ResourceHandle g_cpu_IFFT1D_cplxd64_planner;
        }
    }
}


#endif