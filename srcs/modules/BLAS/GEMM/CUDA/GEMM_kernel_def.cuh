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

#ifndef _GEMM_KERNEL_DEF_CUH_
#define _GEMM_KERNEL_DEF_CUH_

#include "../../../core/basic.h"
#include "../../../DSP/CUDA_cpf32.cuh"


/**
* block(16, 16), ÿ���̴߳��� 8x8=64 �����?(float)��
* ��һ��block����(128, 128)��float���?, �����Ҫÿ���̷߳���?64��32-bit�Ĵ������洢���?
* shmemA -> float4[16][128 / 4]        shmemB -> float4[16][128 / 8]
* һ��ǰ�棬�þ������ˣ�������B�� __linear �����з���, ÿһ���߳�load����float4, ��8��float
* 
* shared memory �ķֲ�ͼ
*   16
* -------                    128
* |     |            -------------------
* |     |            |        B        |
* | A   |    128     |                 |    16
* |     |            -------------------
* |     |
* -------
*/


#define sfma_8x1(name, dex_A, dex_sum) {    \
    sum[dex_sum][0].x = fmaf(tmp_B[0].x, tmp_A[dex_A].name, sum[dex_sum][0].x);        \
    sum[dex_sum][0].y = fmaf(tmp_B[0].y, tmp_A[dex_A].name, sum[dex_sum][0].y);        \
    sum[dex_sum][0].z = fmaf(tmp_B[0].z, tmp_A[dex_A].name, sum[dex_sum][0].z);        \
    sum[dex_sum][0].w = fmaf(tmp_B[0].w, tmp_A[dex_A].name, sum[dex_sum][0].w);        \
    \
    sum[dex_sum][1].x = fmaf(tmp_B[1].x, tmp_A[dex_A].name, sum[dex_sum][1].x);        \
    sum[dex_sum][1].y = fmaf(tmp_B[1].y, tmp_A[dex_A].name, sum[dex_sum][1].y);        \
    sum[dex_sum][1].z = fmaf(tmp_B[1].z, tmp_A[dex_A].name, sum[dex_sum][1].z);        \
    sum[dex_sum][1].w = fmaf(tmp_B[1].w, tmp_A[dex_A].name, sum[dex_sum][1].w);        \
}    \

#define sfma_8x8 {        \
    sfma_8x1(x, 0, 0)    \
    sfma_8x1(y, 0, 1)    \
    sfma_8x1(z, 0, 2)    \
    sfma_8x1(w, 0, 3)    \
\
    sfma_8x1(x, 1, 4)    \
    sfma_8x1(y, 1, 5)    \
    sfma_8x1(z, 1, 6)    \
    sfma_8x1(w, 1, 7)    \
}    \


#define cpl32fma_1x2(name, group_dex, row) {    \
    sum[row]._vf = decx::dsp::fft::GPUK::_complex_2fma1(tmp_B._vf, *((de::CPf*)&tmp_A[group_dex]._vd.name), sum[row]._vf);  \
}


#define cpl32fma_8x8 {        \
    cpl32fma_1x2(x, 0, 0)    \
    cpl32fma_1x2(y, 0, 1)    \
    cpl32fma_1x2(x, 1, 2)    \
    cpl32fma_1x2(y, 1, 3)    \
\
    cpl32fma_1x2(x, 2, 4)    \
    cpl32fma_1x2(y, 2, 5)    \
    cpl32fma_1x2(x, 3, 6)    \
    cpl32fma_1x2(y, 3, 7)    \
}    \


#define s_store_one_line(dex, _dex_name){    \
    dst[_dex_name] = sum[dex][0];            \
    dst[_dex_name + 1] = sum[dex][1];        \
    _dex_name += pitch_B;                  \
}    \




#define s_store(dex_name)  {             \
    s_store_one_line(0, dex_name)        \
    s_store_one_line(1, dex_name)        \
    s_store_one_line(2, dex_name)        \
    s_store_one_line(3, dex_name)        \
    s_store_one_line(4, dex_name)        \
    s_store_one_line(5, dex_name)        \
    s_store_one_line(6, dex_name)        \
    s_store_one_line(7, dex_name)        \
}    \



#define cpl32_store_one_line(dex, _dex_name){    \
    dst[_dex_name] = sum[dex]._vf;         \
    _dex_name += pitch_B;                  \
}    \



#define cpl32_store(dex_name)  {             \
    cpl32_store_one_line(0, dex_name)        \
    cpl32_store_one_line(1, dex_name)        \
    cpl32_store_one_line(2, dex_name)        \
    cpl32_store_one_line(3, dex_name)        \
    cpl32_store_one_line(4, dex_name)        \
    cpl32_store_one_line(5, dex_name)        \
    cpl32_store_one_line(6, dex_name)        \
    cpl32_store_one_line(7, dex_name)        \
}    \



#define _Init_Sum(row){    \
    sum[row][0] = make_float4(0.f, 0.f, 0.f, 0.f);    \
    sum[row][1] = make_float4(0.f, 0.f, 0.f, 0.f);    \
}    \


#define Init_Sum_Union {   \
    sum[0]._vf = make_float4(0.f, 0.f, 0.f, 0.f);   \
    sum[1]._vf = make_float4(0.f, 0.f, 0.f, 0.f);   \
    sum[2]._vf = make_float4(0.f, 0.f, 0.f, 0.f);   \
    sum[3]._vf = make_float4(0.f, 0.f, 0.f, 0.f);   \
    sum[4]._vf = make_float4(0.f, 0.f, 0.f, 0.f);   \
    sum[5]._vf = make_float4(0.f, 0.f, 0.f, 0.f);   \
    sum[6]._vf = make_float4(0.f, 0.f, 0.f, 0.f);   \
    sum[7]._vf = make_float4(0.f, 0.f, 0.f, 0.f);   \
}


#define set_Sum_Union {   \
    sum[0]._vf = make_float4(37.f, 37.f, 37.f, 37.f);   \
    sum[1]._vf = make_float4(37.f, 37.f, 37.f, 37.f);   \
    sum[2]._vf = make_float4(37.f, 37.f, 37.f, 37.f);   \
    sum[3]._vf = make_float4(37.f, 37.f, 37.f, 37.f);   \
    sum[4]._vf = make_float4(37.f, 37.f, 37.f, 37.f);   \
    sum[5]._vf = make_float4(37.f, 37.f, 37.f, 37.f);   \
    sum[6]._vf = make_float4(37.f, 37.f, 37.f, 37.f);   \
    sum[7]._vf = make_float4(37.f, 37.f, 37.f, 37.f);   \
}


#define Init_Sum {    \
    _Init_Sum(0);    \
    _Init_Sum(1);    \
    _Init_Sum(2);    \
    _Init_Sum(3);    \
    _Init_Sum(4);    \
    _Init_Sum(5);    \
    _Init_Sum(6);    \
    _Init_Sum(7);    \
}    \


#if __ABOVE_SM_53

#define hfma_8x1(dex_A) {    \
    fma_tmp.x = tmp_A[dex_A];        \
    fma_tmp.y = fma_tmp.x;            \
    sum[dex_A][0] = __hfma2(fma_tmp, tmp_B[0], sum[dex_A][0]);    \
    sum[dex_A][1] = __hfma2(fma_tmp, tmp_B[1], sum[dex_A][1]);    \
    sum[dex_A][2] = __hfma2(fma_tmp, tmp_B[2], sum[dex_A][2]);    \
    sum[dex_A][3] = __hfma2(fma_tmp, tmp_B[3], sum[dex_A][3]);    \
}    \




#define hfma_8x8 {        \
    hfma_8x1(0);        \
    hfma_8x1(1);        \
    hfma_8x1(2);        \
    hfma_8x1(3);        \
    hfma_8x1(4);        \
    hfma_8x1(5);        \
    hfma_8x1(6);        \
    hfma_8x1(7);        \
}    \


#define h_store_one_line(dex, _dex_name){    \
    dst[_dex_name] = *((float4*)&sum[dex][0]);        \
    _dex_name += pitch_B;    \
}    \



#define h_store(dex_name)  {            \
    h_store_one_line(0, dex_name)        \
    h_store_one_line(1, dex_name)        \
    h_store_one_line(2, dex_name)        \
    h_store_one_line(3, dex_name)        \
    h_store_one_line(4, dex_name)        \
    h_store_one_line(5, dex_name)        \
    h_store_one_line(6, dex_name)        \
    h_store_one_line(7, dex_name)        \
}    \

#endif

#define s_loadC_line(dex, _dex_name){    \
    sum[dex][0] = C[_dex_name];        \
    sum[dex][1] = C[_dex_name + 1];        \
    _dex_name += pitch_B;    \
}    \


#define s_loadC(dex_name)  {        \
    s_loadC_line(0, dex_name)        \
    s_loadC_line(1, dex_name)        \
    s_loadC_line(2, dex_name)        \
    s_loadC_line(3, dex_name)        \
    s_loadC_line(4, dex_name)        \
    s_loadC_line(5, dex_name)        \
    s_loadC_line(6, dex_name)        \
    s_loadC_line(7, dex_name)        \
}    \



#define cpl32_loadC(dex_name) { \
    sum[0]._vf = C[dex_name];       dex_name += pitch_B;    \
    sum[1]._vf = C[dex_name];       dex_name += pitch_B;    \
    sum[2]._vf = C[dex_name];       dex_name += pitch_B;    \
    sum[3]._vf = C[dex_name];       dex_name += pitch_B;    \
    sum[4]._vf = C[dex_name];       dex_name += pitch_B;    \
    sum[5]._vf = C[dex_name];       dex_name += pitch_B;    \
    sum[6]._vf = C[dex_name];       dex_name += pitch_B;    \
    sum[7]._vf = C[dex_name];           \
}



#if __ABOVE_SM_53

#define h_loadC_line(dex, _dex_name){    \
    *((float4*)&sum[dex][0]) = C[_dex_name];        \
    _dex_name += pitch_B;    \
}    \



#define h_loadC(dex_name)  {        \
    h_loadC_line(0, dex_name)        \
    h_loadC_line(1, dex_name)        \
    h_loadC_line(2, dex_name)        \
    h_loadC_line(3, dex_name)        \
    h_loadC_line(4, dex_name)        \
    h_loadC_line(5, dex_name)        \
    h_loadC_line(6, dex_name)        \
    h_loadC_line(7, dex_name)        \
}    \




#define sfma_8x1_mix_accu(row_id){  \
    sum[row_id][0].x = fmaf(__half2float(reg_0[row_id]), __half2float(reg_1[0].x), sum[row_id][0].x);       \
    sum[row_id][0].y = fmaf(__half2float(reg_0[row_id]), __half2float(reg_1[0].y), sum[row_id][0].y);       \
    sum[row_id][0].z = fmaf(__half2float(reg_0[row_id]), __half2float(reg_1[1].x), sum[row_id][0].z);       \
    sum[row_id][0].w = fmaf(__half2float(reg_0[row_id]), __half2float(reg_1[1].y), sum[row_id][0].w);       \
    sum[row_id][1].x = fmaf(__half2float(reg_0[row_id]), __half2float(reg_1[2].x), sum[row_id][1].x);       \
    sum[row_id][1].y = fmaf(__half2float(reg_0[row_id]), __half2float(reg_1[2].y), sum[row_id][1].y);       \
    sum[row_id][1].z = fmaf(__half2float(reg_0[row_id]), __half2float(reg_1[3].x), sum[row_id][1].z);       \
    sum[row_id][1].w = fmaf(__half2float(reg_0[row_id]), __half2float(reg_1[3].y), sum[row_id][1].w);       \
}


#define sfma_8x8_mix_accu {        \
    sfma_8x1_mix_accu(0)     \
    sfma_8x1_mix_accu(1)     \
    sfma_8x1_mix_accu(2)     \
    sfma_8x1_mix_accu(3)     \
\
    sfma_8x1_mix_accu(4)     \
    sfma_8x1_mix_accu(5)     \
    sfma_8x1_mix_accu(6)     \
    sfma_8x1_mix_accu(7)     \
}    \



#define h_cvt_and_store_a_line(row_id){     \
    reg_1[0] = __floats2half2_rn(sum[row_id][0].x, sum[row_id][0].y);     \
    reg_1[1] = __floats2half2_rn(sum[row_id][0].z, sum[row_id][0].w);     \
    reg_1[2] = __floats2half2_rn(sum[row_id][1].x, sum[row_id][1].y);     \
    reg_1[3] = __floats2half2_rn(sum[row_id][1].z, sum[row_id][1].w);     \
    dst[glo_dex_A] = *((float4*)&reg_1);                        \
    glo_dex_A += pitch_B;                                       \
}



#define h_cvt_and_store {   \
    h_cvt_and_store_a_line(0);      \
    h_cvt_and_store_a_line(1);      \
    h_cvt_and_store_a_line(2);      \
    h_cvt_and_store_a_line(3);      \
    h_cvt_and_store_a_line(4);      \
    h_cvt_and_store_a_line(5);      \
    h_cvt_and_store_a_line(6);      \
    h_cvt_and_store_a_line(7);      \
}


#define h_cvt_loadC_a_line(row_id) {     \
*((float4*)&reg_1) = C[glo_dex_A];                                                                \
sum[row_id][0].x = __half2float(reg_1[0].x);     sum[row_id][0].y = __half2float(reg_1[0].y);     \
sum[row_id][0].z = __half2float(reg_1[1].x);     sum[row_id][0].w = __half2float(reg_1[1].y);     \
sum[row_id][1].x = __half2float(reg_1[2].x);     sum[row_id][1].y = __half2float(reg_1[2].y);     \
sum[row_id][1].z = __half2float(reg_1[3].x);     sum[row_id][1].w = __half2float(reg_1[3].y);     \
glo_dex_A += pitch_B;                                                                             \
}



#define h_cvt_loadC {       \
h_cvt_loadC_a_line(0);      \
h_cvt_loadC_a_line(1);      \
h_cvt_loadC_a_line(2);      \
h_cvt_loadC_a_line(3);      \
h_cvt_loadC_a_line(4);      \
h_cvt_loadC_a_line(5);      \
h_cvt_loadC_a_line(6);      \
h_cvt_loadC_a_line(7);      \
}


#endif

#endif