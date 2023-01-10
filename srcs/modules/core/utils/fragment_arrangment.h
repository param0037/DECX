/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _FRAGMENT_MEMAGEMENT_H_
#define _FRAGMENT_MEMAGEMENT_H_

#include "../basic.h"
#include "decx_utils_functions.h"

namespace decx
{
    namespace utils
    {
        struct frag_manager;

        /**
        * @return If can be divided into integer, return true, otherwise return false
        */
        bool frag_manager_gen(decx::utils::frag_manager *src, const size_t _tot, const size_t _frag_num);


        /**
        * @return If can be divided into integer, return true, otherwise return false
        */
        bool frag_manager_gen_from_fragLen(decx::utils::frag_manager* src, const size_t _tot, const size_t _frag_num);


        /**
        * @return If can be divided into integer, return true, otherwise return false
        */
        bool frag_manager_gen_Nx(decx::utils::frag_manager* src, const size_t _tot, const size_t _frag_num, const uint N);
    }
}

struct decx::utils::frag_manager
{
    bool is_left;
    size_t total;               // The total length to be cut
    uint frag_num;            // How many fragments should generate
    uint frag_len;            // The length of each fragment
    uint frag_left_over;      // The length of the leftover fragment (if left)
};



// ----------------------------------------------- 2D ---------------------------------------------------


namespace decx
{
    float _get_ratio_grater_than_one(const uint X, const uint Y);


    uint _get_mid_factors(const uint x);


    uint _get_mid_factors_pow2(const uint x);


    uint2 find_closest_factors_by_ratio(const uint x, const uint2 pair);


    uint2 find_closest_factors_by_ratio_pow2(const uint x, const uint2 pair, bool pow2_on_x);


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
        void thread2D_arrangement_advisor_for_GEMM(uint2* thr_arrange, const uint total_thr_num, const uint2 proc_dims);
    }
}



#endif