/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GPU_CONV2_UTILS_CUH_
#define _GPU_CONV2_UTILS_CUH_


namespace decx {
    namespace conv_I2R
    {
        struct _conv2_I2C_params_set
        {
            int2 ker_dims, ker_buf_dim;
            uint Wsrc_buf, WI2C_buf, depth, src_proc_H, Wdst_eqMM, k_tensor_num, Wdst_o;
        };
    }
}


#endif