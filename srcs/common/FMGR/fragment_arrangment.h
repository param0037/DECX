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


#ifndef _FRAGMENT_MEMAGEMENT_H_
#define _FRAGMENT_MEMAGEMENT_H_

#include "../basic.h"
#include "../decx_utils_functions.h"


namespace decx{
namespace utils {
    template <typename _addr_type>
    struct unpitched_frac_mapping;
}
}



template <typename _addr_type>
struct decx::utils::unpitched_frac_mapping
{
    uint32_t _pitch_L1;
    uint32_t _effective_L1;

    uint32_t _pitch_L2;
    uint32_t _effective_L2;

    bool _is_Level2;
    bool _is_zipped;

#ifdef _DECX_CUDA_PARTS_
    __host__ __device__
#endif
    unpitched_frac_mapping() {}


#ifdef _DECX_CUDA_PARTS_
    __host__ __device__
#endif
    void set_attributes(const uint32_t pitch, const uint32_t effective)
    {
        this->_pitch_L1 = pitch;
        this->_effective_L1 = effective;

        this->_pitch_L2 = 0;
        this->_effective_L2 = 0;

        this->_is_Level2 = false;
        this->_is_zipped = (this->_pitch_L1 != this->_effective_L1);
    }


#ifdef _DECX_CUDA_PARTS_
    __host__ __device__
#endif
    void set_attributes(const uint32_t pitch_L1, const uint32_t effective_L1, const uint32_t pitch_L2, const uint32_t effective_L2)
    {
        this->_pitch_L1 = pitch_L1;
        this->_effective_L1 = effective_L1;

        this->_pitch_L2 = pitch_L2 * pitch_L1;
        this->_effective_L2 = effective_L2 * effective_L1;

        this->_is_Level2 = true;
        this->_is_zipped = this->_is_zipped = (this->_pitch_L1 != this->_effective_L1 || this->_pitch_L2 != this->_effective_L2);
    }


#ifdef _DECX_CUDA_PARTS_
    __host__ __device__
#endif
    __inline _addr_type get_phyaddr_L1(const _addr_type _viraddr) const
    {
        return (_viraddr / this->_effective_L1) * this->_pitch_L1 + (_viraddr % this->_effective_L1);
    }


#ifdef _DECX_CUDA_PARTS_
    __host__ __device__
#endif
    __inline _addr_type get_phyaddr_L2(const _addr_type _viraddr) const
    {
        return (_viraddr / this->_effective_L2) * this->_pitch_L2 + this->get_phyaddr_L1(_viraddr % this->_effective_L2);
    }
};



namespace decx
{
    namespace utils
    {
        struct frag_manager;

        /**
        * Ensuring that frag_manager::frag_num = _frag_num, but, frag_manager::frag_len depends on _tot
        * @return If can be divided into integer, return true, otherwise return false
        */
        _DECX_API_ bool frag_manager_gen(decx::utils::frag_manager *src, const uint64_t _tot, const uint64_t _frag_num);


        /**
        * Ensuring that frag_manager::frag_len = _frag_len, but, frag_manager::frag_num depends on _tot
        * @return If can be divided into integer, return true, otherwise return false
        */
        _DECX_API_ bool frag_manager_gen_from_fragLen(decx::utils::frag_manager* src, const uint64_t _tot, const uint64_t _frag_len);


        /**
        * Ensuring that frag_manager::frag_len can be divided by N into integer, but, frag_manager::frag_left_over might not
        * @return If can be divided into integer, return true, otherwise return false
        */
        _DECX_API_ bool frag_manager_gen_Nx(decx::utils::frag_manager* src, const uint64_t _tot, const uint64_t _frag_num, const uint N);
    }
}


struct decx::utils::frag_manager
{
    bool is_left;
    uint64_t total;                 // The total length to be cut
    uint32_t frag_num;              // How many fragments should generate
    uint32_t frag_len;              // The length of each fragment
    uint32_t frag_left_over;        // The length of the leftover fragment (if left)
    uint32_t last_frag_len;         // The length of the very last fragment
};



// ----------------------------------------------- 2D ---------------------------------------------------


namespace decx
{
namespace utils
{
    /**
    * @brief This function arrange threads for a 2D matrix.
    *
    * @param thr_arrange : The two dimensions of processed 2D area
    * @param total_thr_num The number of total thread
    * @param proc_dims : .x -> width; .y ->height
    *
    * @return The return value is a uint2 structure, of which x and y are propotional to proc_dims
    */
    void thread2D_arrangement_advisor(uint2* thr_arrange, const uint32_t total_thr_num, const uint2 proc_dims);


    typedef struct _blocking2D_fmgrs_t {
        decx::utils::frag_manager _fmgrH;
        decx::utils::frag_manager _fmgrW;
    }_blocking2D_fmgrs;
}
}



#endif