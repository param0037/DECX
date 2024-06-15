/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../GEMM_utils.h"

template <>
void decx::cpu_GEMM_general_config<float>::Config(const uint32_t parallel, 
                                                  const decx::_matrix_layout* layout_A, 
                                                  const decx::_matrix_layout* layout_B,
                                                  de::DH* handle)
{
    this->_layout_A = layout_A;
    this->_layout_B = layout_B;
    this->_concurrency = parallel;

    this->_arranged_B_dims = make_uint2(this->_layout_A->pitch * 16, 
                                        decx::utils::ceil<uint32_t>(this->_layout_B->pitch, 16));     // assume that it is 16x

    if (decx::alloc::_host_virtual_page_malloc(
        &this->_arranged_B, this->_arranged_B_dims.x * this->_arranged_B_dims.y * sizeof(float), true)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }

    this->_L = this->_layout_B->pitch % 16;
    decx::utils::thread2D_arrangement_advisor(&this->_thread_dist, 
                                              this->_concurrency,
                                              make_uint2(this->_layout_B->pitch, this->_layout_A->height));
    this->_t2D.reshape(this->_thread_dist.y, this->_thread_dist.x);

    if (this->_L) {
        /* The model of _B->Pitch() is 16N + 8.
         * In the program below, fullfilling the pitch is used, which is 16N + 8 + 8 = 16(N+1) */
        decx::utils::frag_manager_gen(&this->_pre_f_mgrW, (this->_layout_B->pitch + 8) / 16, this->_t2D.thread_w);

        // assign value for the pitch of extra_cache_dst, which is the proc_w of the thread that process the end of a row
        this->_pitch_extdst = this->_pre_f_mgrW.is_left ? this->_pre_f_mgrW.frag_left_over : this->_pre_f_mgrW.frag_len;

        // generate the configuration for sorting matrix B
        decx::utils::frag_manager_gen(&this->_f_mgr_sort_B, (this->_layout_B->pitch + 8) / 16, this->_concurrency);

        if (decx::alloc::_host_virtual_page_malloc(&this->_extra_dst, this->_pitch_extdst * this->_layout_A->height * 16 * sizeof(float), true)) {
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
                ALLOC_FAIL);
            return;
        }
    }
    else {
        this->_pitch_extdst = 0;
        // generate the configuration for sorting matrix B
        decx::utils::frag_manager_gen(&this->_f_mgr_sort_B, this->_arranged_B_dims.y, this->_t2D.total_thread);
    }
}


template <typename _data_type>
bool decx::cpu_GEMM_general_config<_data_type>::Changed(const uint32_t parallel, 
                                                        const decx::_matrix_layout* layout_A,
                                                        const decx::_matrix_layout* layout_B) const
{
    uint32_t Adims_cmp, Bdims_cmp;
    if (this->_layout_A){
        Adims_cmp = (this->_layout_A->height ^ layout_A->height) | 
                                   (this->_layout_A->width ^ layout_A->width) |
                                   (this->_layout_A->pitch ^ layout_A->pitch);
    }
    else { Adims_cmp = 1; }

    if (this->_layout_B) {
        Bdims_cmp = (this->_layout_B->height ^ layout_B->height) | 
                                   (this->_layout_B->width ^ layout_B->width) |
                                   (this->_layout_B->pitch ^ layout_B->pitch);
    }
    else { Bdims_cmp = 1; }

    const uint32_t conc_cmp = this->_concurrency ^ parallel;
    return (Adims_cmp | Bdims_cmp | conc_cmp);
}

template bool decx::cpu_GEMM_general_config<float>::Changed(const uint32_t, const decx::_matrix_layout*, const decx::_matrix_layout*) const;
