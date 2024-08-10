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

#ifndef _MMA_FP32_CUH_
#define _MMA_FP32_CUH_


#define _DECL_A_DEX_OUTER_FP32_(_row_id) (_row_id >> 2)
#define _DECL_A_DEX_INNER_FP32_(_row_id) (_row_id & 3)
#define _DECL_A_ARG_FP32_(regsA, _row_id) (regsA[_DECL_A_DEX_OUTER_FP32_(_row_id)]._arrf[_DECL_A_DEX_INNER_FP32_(_row_id)])


#define _MMA_FP32_1_4_1_(regsA, regsB, accu, _row_id) {   \
    accu[_row_id]._vf.x = __fmaf_rn(_DECL_A_ARG_FP32_(regsA, _row_id), regsB._vf.x, accu[_row_id]._vf.x);    \
    accu[_row_id]._vf.y = __fmaf_rn(_DECL_A_ARG_FP32_(regsA, _row_id), regsB._vf.y, accu[_row_id]._vf.y);    \
    accu[_row_id]._vf.z = __fmaf_rn(_DECL_A_ARG_FP32_(regsA, _row_id), regsB._vf.z, accu[_row_id]._vf.z);    \
    accu[_row_id]._vf.w = __fmaf_rn(_DECL_A_ARG_FP32_(regsA, _row_id), regsB._vf.w, accu[_row_id]._vf.w);    \
}


#define _MMA_FP32_1_4_16_(regsA, regsB1, accu) {        \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 0);      \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 1);      \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 2);      \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 3);      \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 4);      \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 5);      \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 6);      \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 7);      \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 8);      \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 9);      \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 10);     \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 11);     \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 12);     \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 13);     \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 14);     \
    _MMA_FP32_1_4_1_(regsA, regsB1, accu, 15);     \
}



#endif
