/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _CONV2_MC_MACROS_H_
#define _CONV2_MC_MACROS_H_

#include "../../../core/cudaStream_management/cudaStream_queue.h"

#define conv3_kernel(kernel_name_1, kernel_name_2) {        \
if (Dmem1->leading) {                                       \
                                                            \
    kernel_name_1;                                          \
    Dmem1->_using = true;                                   \
    Dmem3->_using = true;                                   \
                                                            \
    Dmem3->leading = true;                                  \
    Dmem4->leading = false;                                 \
}                                                           \
else {                                                      \
                                                            \
    kernel_name_2;                                          \
    Dmem2->_using = true;                                   \
    Dmem4->_using = true;                                   \
                                                            \
    Dmem4->leading = true;                                  \
    Dmem3->leading = false;                                 \
}                                                           \
}


#define conv3_main_loop_MemCpyHtoD_within_NB(__type, __ele_num) {                                               \
if (!Dmem1->_using) {                                                                                           \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                          \
        reinterpret_cast<__type*>(Dmem1->mem) + src_diff.x * Dsrc_alloc_dim->x * __ele_num + src_diff.y,        \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),             \
        src->width * sizeof(__type), src->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));                               \
                                                                                                                \
    Dmem1->leading = true;                                                                                      \
    Dmem2->leading = false;                                                                                     \
}                                                                                                               \
else {                                                                                                          \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                          \
        reinterpret_cast<__type*>(Dmem2->mem) + src_diff.x * Dsrc_alloc_dim->x * __ele_num + src_diff.y,        \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),             \
        src->width * sizeof(__type), src->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));                               \
                                                                                                                \
    Dmem2->leading = true;                                                                                      \
    Dmem1->leading = false;                                                                                     \
}                                                                                                               \
}



#define conv3_main_loop_MemCpyDtoD_within_NB(__type, __ele_num) {                                            \
if (!Dmem1->_using) {                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                        \
        reinterpret_cast<__type*>(Dmem1->mem) + src_diff.x * Dsrc_alloc_dim->x * __ele_num + src_diff.y,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),            \
        src->width * sizeof(__type), src->height, cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref()));                            \
                                                                                                            \
    Dmem1->leading = true;                                                                                    \
    Dmem2->leading = false;                                                                                    \
}                                                                                                            \
else {                                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                        \
        reinterpret_cast<__type*>(Dmem2->mem) + src_diff.x * Dsrc_alloc_dim->x * __ele_num + src_diff.y,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),            \
        src->width * sizeof(__type), src->height, cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref()));                            \
                                                                                                            \
    Dmem2->leading = true;                                                                                    \
    Dmem1->leading = false;                                                                                    \
}                                                                                                            \
}



#define conv3_main_loop_MemCpyHtoD_exact_NB(__type) {                                                        \
if (!Dmem1->_using) {                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                        \
        Dmem1->mem,                                                                                            \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),            \
        src->width * sizeof(__type), src->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));                            \
                                                                                                            \
    Dmem1->leading = true;                                                                                    \
    Dmem2->leading = false;                                                                                    \
}                                                                                                            \
else {                                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                        \
        Dmem2->mem,                                                                                            \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),            \
        src->width * sizeof(__type), src->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));                            \
                                                                                                            \
    Dmem2->leading = true;                                                                                    \
    Dmem1->leading = false;                                                                                    \
}                                                                                                            \
}



#define conv3_main_loop_MemCpyDtoD_exact_NB(__type) {                                                        \
if (!Dmem1->_using) {                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                        \
        Dmem1->mem,                                                                                            \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),            \
        src->width * sizeof(__type), src->height, cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref()));                            \
                                                                                                            \
    Dmem1->leading = true;                                                                                    \
    Dmem2->leading = false;                                                                                    \
}                                                                                                            \
else {                                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                        \
        Dmem2->mem,                                                                                            \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),            \
        src->width * sizeof(__type), src->height, cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref()));                            \
                                                                                                            \
    Dmem2->leading = true;                                                                                    \
    Dmem1->leading = false;                                                                                    \
}                                                                                                            \
}



/**
* for the host_to_device memory copy in border_constant term
* @param _Wker_preset : the preset kernel width
* @param _Hker_preset : the preset kernel height
*/
#define conv3_main_loop_MemCpyHtoD_BC(_Hker_preset, _Wker_preset, _type, _ele_num) {                        \
if (!Dmem1->_using) {                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                        \
        reinterpret_cast<_type*>(Dmem1->mem) + _Hker_preset * Dsrc_alloc_dim->x * _ele_num + _Wker_preset,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(_type),            \
        src->width * sizeof(_type), src->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));                            \
                                                                                                            \
    Dmem1->leading = true;                                                                                    \
    Dmem2->leading = false;                                                                                    \
}                                                                                                            \
else {                                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                        \
        reinterpret_cast<_type*>(Dmem2->mem) + _Hker_preset * Dsrc_alloc_dim->x * _ele_num + _Wker_preset,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(_type),            \
        src->width * sizeof(_type), src->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));                            \
                                                                                                            \
    Dmem2->leading = true;                                                                                    \
    Dmem1->leading = false;                                                                                    \
}                                                                                                            \
}


/**
* for the host_to_device memory copy in border_constant term
* @param _Wker_preset : the preset kernel width
* @param _Hker_preset : the preset kernel height
*/
#define conv3_main_loop_MemCpyDtoD_BC(_Hker_preset, _Wker_preset, _type, _ele_num) {                        \
if (!Dmem1->_using) {                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                        \
        reinterpret_cast<_type*>(Dmem1->mem) + _Hker_preset * Dsrc_alloc_dim->x * _ele_num + _Wker_preset,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(_type),            \
        src->width * sizeof(_type), src->height, cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref()));                            \
                                                                                                            \
    Dmem1->leading = true;                                                                                    \
    Dmem2->leading = false;                                                                                    \
}                                                                                                            \
else {                                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                        \
        reinterpret_cast<_type*>(Dmem2->mem) + _Hker_preset * Dsrc_alloc_dim->x * _ele_num + _Wker_preset,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(_type),            \
        src->width * sizeof(_type), src->height, cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref()));                            \
                                                                                                            \
    Dmem2->leading = true;                                                                                    \
    Dmem1->leading = false;                                                                                    \
}                                                                                                            \
}


// ----------------------------------------- Multi-Kernel -----------------------------------------------


#define conv3_main_loop_MemCpyHtoD_within_NB_MK(__type, __ele_num) {                                                    \
for (int k = 0; k < kernel->height; ++k) {                                                                                \
    cudaMemcpyToSymbolAsync(Const_Mem, (__type*)kernel->MatptrArr.ptr[i + 1] + offset_ker,                                        \
        kernel->width * sizeof(__type), sym_cpy_offset + offset_lin * sizeof(__type), cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref());    \
    offset_lin += kernel->width;                                                                                        \
    offset_ker += kernel->pitch;                                                                                        \
}                                                                                                                        \
offset_lin = 0;        offset_ker = 0;                                                                                        \
if (!Dmem1->_using) {                                                                                                    \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                                    \
        reinterpret_cast<__type*>(Dmem1->mem) + src_diff.x * Dsrc_alloc_dim->x * __ele_num + src_diff.y,                \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),                        \
        src->width * sizeof(__type), src->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));                                        \
                                                                                                                        \
    Dmem1->leading = true;                                                                                                \
    Dmem2->leading = false;                                                                                                \
}                                                                                                                        \
else {                                                                                                                    \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                                    \
        reinterpret_cast<__type*>(Dmem2->mem) + src_diff.x * Dsrc_alloc_dim->x * __ele_num + src_diff.y,                \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),                        \
        src->width * sizeof(__type), src->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));                                        \
                                                                                                                        \
    Dmem2->leading = true;                                                                                                \
    Dmem1->leading = false;                                                                                                \
}                                                                                                                        \
}



#define conv3_main_loop_MemCpyDtoD_within_NB_MK(__type, __ele_num) {                                                    \
for (int k = 0; k < kernel->height; ++k) {                                                                                \
    cudaMemcpyToSymbolAsync(Const_Mem, kernel->MatptrArr.ptr[i + 1] + offset_ker,                                        \
        kernel->width * sizeof(__type), sym_cpy_offset + offset_lin * sizeof(__type), cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref());    \
    offset_lin += kernel->width;                                                                                        \
    offset_ker += kernel->pitch;                                                                                        \
}                                                                                                                        \
offset_lin = 0;        offset_ker = 0;                                                                                        \
if (!Dmem1->_using) {                                                                                                    \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                                    \
        reinterpret_cast<__type*>(Dmem1->mem) + src_diff.x * Dsrc_alloc_dim->x * __ele_num + src_diff.y,                \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),                        \
        src->width * sizeof(__type), src->height, cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref()));                                        \
                                                                                                                        \
    Dmem1->leading = true;                                                                                                \
    Dmem2->leading = false;                                                                                                \
}                                                                                                                        \
else {                                                                                                                    \
    checkCudaErrors(cudaMemcpy2DAsync(                                                                                    \
        reinterpret_cast<__type*>(Dmem2->mem) + src_diff.x * Dsrc_alloc_dim->x * __ele_num + src_diff.y,                \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),                        \
        src->width * sizeof(__type), src->height, cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref()));                                        \
                                                                                                                        \
    Dmem2->leading = true;                                                                                                \
    Dmem1->leading = false;                                                                                                \
}                                                                                                                        \
}



#define conv3_main_loop_MemCpyHtoD_exact_NB_MK(__type) {    \
for (int k = 0; k < kernel->height; ++k) {    \
    cudaMemcpyToSymbolAsync(Const_Mem, (__type*)kernel->MatptrArr.ptr[i + 1] + offset_ker,    \
        kernel->width * sizeof(__type), sym_cpy_offset + offset_lin * sizeof(__type), cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref());    \
    offset_lin += kernel->width;    \
    offset_ker += kernel->pitch;    \
}    \
offset_lin = 0;        offset_ker = 0;    \
if (!Dmem1->_using) {    \
    checkCudaErrors(cudaMemcpy2DAsync(        \
        Dmem1->mem,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),    \
        src->width * sizeof(__type), src->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));    \
\
    Dmem1->leading = true;    \
    Dmem2->leading = false;    \
\
    Dmem1->leading = true;    \
    Dmem2->leading = false;    \
}    \
else {    \
    checkCudaErrors(cudaMemcpy2DAsync(        \
        Dmem2->mem,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),    \
        src->width * sizeof(__type), src->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));    \
\
    Dmem2->leading = true;    \
    Dmem1->leading = false;    \
}    \
}



#define conv3_main_loop_MemCpyDtoD_exact_NB_MK(__type) {    \
for (int k = 0; k < kernel->height; ++k) {    \
    cudaMemcpyToSymbolAsync(Const_Mem, kernel->MatptrArr.ptr[i + 1] + offset_ker,    \
        kernel->width * sizeof(__type), sym_cpy_offset + offset_lin * sizeof(__type), cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref());    \
    offset_lin += kernel->width;    \
    offset_ker += kernel->pitch;    \
}    \
offset_lin = 0;        offset_ker = 0;    \
if (!Dmem1->_using) {    \
    checkCudaErrors(cudaMemcpy2DAsync(        \
        Dmem1->mem,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),    \
        src->width * sizeof(__type), src->height, cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref()));    \
\
    Dmem1->leading = true;    \
    Dmem2->leading = false;    \
\
    Dmem1->leading = true;    \
    Dmem2->leading = false;    \
}    \
else {    \
    checkCudaErrors(cudaMemcpy2DAsync(        \
        Dmem2->mem,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(__type),    \
        src->width * sizeof(__type), src->height, cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref()));    \
\
    Dmem2->leading = true;    \
    Dmem1->leading = false;    \
}    \
}



/**
* for the host_to_device memory copy in border_constant term
* @param _Wker_preset : the preset kernel width
* @param _Hker_preset : the preset kernel height
*/
#define conv3_main_loop_MemCpyHtoD_BC_MK(_Hker_preset, _Wker_preset, _type, _ele_num) {    \
for (int k = 0; k < kernel->height; ++k) {    \
    cudaMemcpyToSymbolAsync(Const_Mem, (_type*)kernel->MatptrArr.ptr[i + 1] + offset_ker,    \
        kernel->width * sizeof(_type), sym_cpy_offset + offset_lin * sizeof(_type), cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref());    \
    offset_lin += kernel->width;    \
    offset_ker += kernel->pitch;    \
}    \
offset_lin = 0;        offset_ker = 0;    \
if (!Dmem1->_using) {    \
    checkCudaErrors(cudaMemcpy2DAsync(        \
        reinterpret_cast<_type*>(Dmem1->mem) + _Hker_preset * Dsrc_alloc_dim->x * _ele_num + _Wker_preset,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(_type),    \
        src->width * sizeof(_type), src->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));    \
\
    Dmem1->leading = true;    \
    Dmem2->leading = false;    \
\
    Dmem1->leading = true;    \
    Dmem2->leading = false;    \
}    \
else {    \
    checkCudaErrors(cudaMemcpy2DAsync(        \
        reinterpret_cast<_type*>(Dmem2->mem) + _Hker_preset * Dsrc_alloc_dim->x * _ele_num + _Wker_preset,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(_type),    \
        src->width * sizeof(_type), src->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));    \
\
    Dmem2->leading = true;    \
    Dmem1->leading = false;    \
}    \
}



/**
* for the host_to_device memory copy in border_constant term
* @param _Wker_preset : the preset kernel width
* @param _Hker_preset : the preset kernel height
*/
#define conv3_main_loop_MemCpyDtoD_BC_MK(_Hker_preset, _Wker_preset, _type, _ele_num) {    \
for (int k = 0; k < kernel->height; ++k) {    \
    cudaMemcpyToSymbolAsync(Const_Mem, kernel->MatptrArr.ptr[i + 1] + offset_ker,    \
        kernel->width * sizeof(_type), sym_cpy_offset + offset_lin * sizeof(_type), cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref());    \
    offset_lin += kernel->width;    \
    offset_ker += kernel->pitch;    \
}    \
offset_lin = 0;        offset_ker = 0;    \
if (!Dmem1->_using) {    \
    checkCudaErrors(cudaMemcpy2DAsync(        \
        reinterpret_cast<_type*>(Dmem1->mem) + _Hker_preset * Dsrc_alloc_dim->x * _ele_num + _Wker_preset,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(_type),    \
        src->width * sizeof(_type), src->height, cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref()));    \
\
    Dmem1->leading = true;    \
    Dmem2->leading = false;    \
\
    Dmem1->leading = true;    \
    Dmem2->leading = false;    \
}    \
else {    \
    checkCudaErrors(cudaMemcpy2DAsync(        \
        reinterpret_cast<_type*>(Dmem2->mem) + _Hker_preset * Dsrc_alloc_dim->x * _ele_num + _Wker_preset,    \
        Dsrc_alloc_dim->x * sizeof(float4), src->MatptrArr.ptr[i + 1], src->pitch * sizeof(_type),    \
        src->width * sizeof(_type), src->height, cudaMemcpyDeviceToDevice, S[1]->get_raw_stream_ref()));    \
\
    Dmem2->leading = true;    \
    Dmem1->leading = false;    \
}    \
}





#define main_loop_regulable_R_sk(kernel_name_1, kernel_name_2, memcpy_name, _type) {                                        \
                                                                                                                            \
for (int i = 0; i < src->ArrayNumber; ++i){                                                                                    \
    if (i > 0) {                                                                                                            \
        if (Dmem3->leading) {                                                                                                \
            checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i - 1],                                                    \
                dst->pitch * sizeof(_type), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(_type),        \
                dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));                                                                \
            Dmem3->_using = true;                                                                                            \
        }                                                                                                                    \
        else {                                                                                                                \
            checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i - 1],                                                    \
                dst->pitch * sizeof(_type), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(_type),        \
                dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));                                                                \
            Dmem4->_using = true;                                                                                            \
        }                                                                                                                    \
    }                                                                                                                        \
    conv3_kernel(kernel_name_1, kernel_name_2)                                                                                \
    if (i < src->ArrayNumber - 1) {                                                                                            \
        memcpy_name                                                                                                            \
    }                                                                                                                        \
                                                                                                                            \
    checkCudaErrors(cudaDeviceSynchronize());                                                                                \
                                                                                                                            \
    Dmem1->_using = false;                                                                                                    \
    Dmem3->_using = false;                                                                                                    \
                                                                                                                            \
    Dmem2->_using = false;                                                                                                    \
    Dmem4->_using = false;                                                                                                    \
}                                                                                                                            \
}                                                                                                                            \





#define main_loop_regulable_R_mk(kernel_name_1, kernel_name_2, memcpy_name, _type) {    \
\
for (int i = 0; i < src->ArrayNumber; ++i){        \
    if (i > 0) {    \
        if (Dmem3->leading) {    \
            checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i - 1],    \
                dst->pitch * sizeof(_type), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(_type),    \
                dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));    \
            Dmem3->_using = true;    \
        }    \
        else {    \
            checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[i - 1],    \
                dst->pitch * sizeof(_type), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(_type),    \
                dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));    \
            Dmem4->_using = true;    \
        }    \
    }    \
    conv3_kernel(kernel_name_1, kernel_name_2)    \
\
    if (rep_const_mem0.leading){sym_cpy_offset = CM_offset;    decx::utils::set_mutex_memory_state<void>(&rep_const_mem1, &rep_const_mem0);}    \
    else{sym_cpy_offset = 0;    decx::utils::set_mutex_memory_state<void>(&rep_const_mem0, &rep_const_mem1);}    \
\
    if (i < src->ArrayNumber - 1) {    \
        memcpy_name    \
    }    \
        \
    checkCudaErrors(cudaDeviceSynchronize());    \
        \
    Dmem1->_using = false;    \
    Dmem3->_using = false;    \
        \
    Dmem2->_using = false;    \
    Dmem4->_using = false;    \
}    \
}    \




#endif