/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "conv_utils.h"


decx::_Conv2_MK_Props_fp32::_Conv2_MK_Props_fp32(const uint2                 _ker_dims,
                                                 const uint                  _W_original_src,
                                                 const uint                  _W_tmp_src,
                                                 const size_t                _page_size,
                                                 const uint                  _channel_size,
                                                 decx::utils::frag_manager* _f_mgr,
                                                 const size_t                _page_size_src,      // set only when non-border conv2d occurs
                                                 const size_t                _page_size_ker       // set only when multi-kernel conv2d occurs
        ) :
            ker_dims(_ker_dims),
            Wdst(_W_original_src),
            Wsrc(_W_tmp_src),
            page_size_dst(_page_size),
            page_size_src(_page_size_src),
            page_size_ker(_page_size_ker),
            channel_size(_channel_size),
            f_mgr(_f_mgr)
{
    const uint half_kernel_w = ker_dims.x / 2;
    if (half_kernel_w < 5) {
        this->_loop = 0;
        this->reg_WL = 0;
    }
    else {
        this->_loop = (uint)decx::utils::clamp_min<int>((((int)this->ker_dims.x / 2) - 5) / 4, 0);
        this->reg_WL = (ushort)(this->ker_dims.x - 1 - 8 - _loop * 8);
    }
}




decx::_Conv2_MK_Props_fp64::_Conv2_MK_Props_fp64(const uint2                 _ker_dims,
                             const uint                  _W_original_src, 
                             const uint                  _W_tmp_src, 
                             const size_t                _page_size, 
                             const uint                  _channel_size,
                             decx::utils::frag_manager*  _f_mgr,
                             const size_t                _page_size_src,      // set only when non-border conv2d occurs
                             const size_t                _page_size_ker       // set only when multi-kernel conv2d occurs
        ) :
            ker_dims(_ker_dims), 
            Wdst(_W_original_src),
            Wsrc(_W_tmp_src),
            page_size_dst(_page_size),
            page_size_src(_page_size_src),
            page_size_ker(_page_size_ker),
            channel_size(_channel_size),
            f_mgr(_f_mgr)
{
    const uint half_kernel_w = ker_dims.x / 2;
    if (half_kernel_w < 5) {
        this->_loop = 0;
        this->reg_WL = 0;
    }
    else {
        this->_loop = (uint)decx::utils::clamp_min<int>((((int)this->ker_dims.x / 2) - 3) / 2, 0);
        this->reg_WL = (ushort)(this->ker_dims.x - 1 - 4 - _loop * 4);
    }
}