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