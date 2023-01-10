/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _CONV2_BORDER_CONST_MK_FP16_H_
#define _CONV2_BORDER_CONST_MK_FP16_H_


#include "../../../../core/basic.h"
#include "../Conv2_MC_macros.h"


namespace decx
{
    static void main_loop_hconv2_mk_within8x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3], const int flag);


    static void main_loop_hconv2_mk_exact8x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3], const int flag);


    static void main_loop_hconv2_mk_within8x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3], const int flag);


    static void main_loop_hconv2_mk_exact8x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3], const int flag);


    static void main_loop_hconv2_mk_within16x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3], const int flag);


    static void main_loop_hconv2_mk_exact16x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3], const int flag);


    static void main_loop_hconv2_mk_within16x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3], const int flag);


    static void main_loop_hconv2_mk_exact16x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3], const int flag);
}





static void decx::main_loop_hconv2_mk_within8x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim,
        int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
    decx::cuda_stream* S[3], const int flag)
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R8 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim->x / 2;

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 8 + bounded_kernel_R8,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    size_t offset_lin = 0, offset_ker = 0;
    // strat the main loop
    MIF<void> rep_const_mem0, rep_const_mem1;
    decx::utils::set_mutex_memory_state<void>(&rep_const_mem0, &rep_const_mem1);
    const size_t CM_offset = kernel->width * kernel->height * sizeof(de::Half);
    size_t sym_cpy_offset = 0;

    main_loop_regulable_R_mk(
        hconv2_kernel_within8x8_offset(Dmem1->mem, Dmem3->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        hconv2_kernel_within8x8_offset(Dmem2->mem, Dmem4->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        conv3_main_loop_MemCpyHtoD_BC_MK(bounded_kernel_R8, bounded_kernel_R8, de::Half, 8),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}



static void decx::main_loop_hconv2_mk_exact8x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim,
        int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
    decx::cuda_stream* S[3], const int flag)
{
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 8 + bounded_kernel_R8,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    size_t offset_lin = 0, offset_ker = 0;
    // strat the main loop
    MIF<void> rep_const_mem0, rep_const_mem1;
    decx::utils::set_mutex_memory_state<void>(&rep_const_mem0, &rep_const_mem1);
    const size_t CM_offset = kernel->width * kernel->height * sizeof(de::Half);
    size_t sym_cpy_offset = 0;

    main_loop_regulable_R_mk(
        hconv2_kernel_exact8x8_offset(Dmem1->mem, Dmem3->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        hconv2_kernel_exact8x8_offset(Dmem2->mem, Dmem4->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        conv3_main_loop_MemCpyHtoD_BC_MK(bounded_kernel_R8, bounded_kernel_R8, de::Half, 8),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_hconv2_mk_within8x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim,
        int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
    decx::cuda_stream* S[3], const int flag)
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R8 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim->x / 2;

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 8 + bounded_kernel_R16,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    size_t offset_lin = 0, offset_ker = 0;
    // strat the main loop
    MIF<void> rep_const_mem0, rep_const_mem1;
    decx::utils::set_mutex_memory_state<void>(&rep_const_mem0, &rep_const_mem1);
    const size_t CM_offset = kernel->width * kernel->height * sizeof(de::Half);
    size_t sym_cpy_offset = 0;

    main_loop_regulable_R_mk(
        hconv2_kernel_within8x16_offset(Dmem1->mem, Dmem3->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        hconv2_kernel_within8x16_offset(Dmem2->mem, Dmem4->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        conv3_main_loop_MemCpyHtoD_BC_MK(bounded_kernel_R8, bounded_kernel_R16, de::Half, 8),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_hconv2_mk_exact8x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim,
        int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
    decx::cuda_stream* S[3], const int flag)
{
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 8 + bounded_kernel_R16,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    size_t offset_lin = 0, offset_ker = 0;
    // strat the main loop
    MIF<void> rep_const_mem0, rep_const_mem1;
    decx::utils::set_mutex_memory_state<void>(&rep_const_mem0, &rep_const_mem1);
    const size_t CM_offset = kernel->width * kernel->height * sizeof(de::Half);
    size_t sym_cpy_offset = 0;

    main_loop_regulable_R_mk(
        decx::hconv2_kernel_exact8x16_offset(Dmem1->mem, Dmem3->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        decx::hconv2_kernel_exact8x16_offset(Dmem2->mem, Dmem4->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        conv3_main_loop_MemCpyHtoD_BC_MK(bounded_kernel_R8, bounded_kernel_R16, de::Half, 8),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_hconv2_mk_within16x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim,
        int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
    decx::cuda_stream* S[3], const int flag)
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R16 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim->x / 2;

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 8 + bounded_kernel_R8,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    size_t offset_lin = 0, offset_ker = 0;
    // strat the main loop
    MIF<void> rep_const_mem0, rep_const_mem1;
    decx::utils::set_mutex_memory_state<void>(&rep_const_mem0, &rep_const_mem1);
    const size_t CM_offset = kernel->width * kernel->height * sizeof(de::Half);
    size_t sym_cpy_offset = 0;

    main_loop_regulable_R_mk(
        hconv2_kernel_within16x8_offset(Dmem1->mem, Dmem3->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        hconv2_kernel_within16x8_offset(Dmem2->mem, Dmem4->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        conv3_main_loop_MemCpyHtoD_BC_MK(bounded_kernel_R16, bounded_kernel_R8, de::Half, 8),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_hconv2_mk_exact16x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim,
        int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
    decx::cuda_stream* S[3], const int flag)
{
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 8 + bounded_kernel_R8,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    size_t offset_lin = 0, offset_ker = 0;
    // strat the main loop
    MIF<void> rep_const_mem0, rep_const_mem1;
    decx::utils::set_mutex_memory_state<void>(&rep_const_mem0, &rep_const_mem1);
    const size_t CM_offset = kernel->width * kernel->height * sizeof(de::Half);
    size_t sym_cpy_offset = 0;

    main_loop_regulable_R_mk(
        hconv2_kernel_exact16x8_offset(Dmem1->mem, Dmem3->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        hconv2_kernel_exact16x8_offset(Dmem2->mem, Dmem4->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        conv3_main_loop_MemCpyHtoD_BC_MK(bounded_kernel_R16, bounded_kernel_R8, de::Half, 8),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_hconv2_mk_within16x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim,
        int2* ker_dim,
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
    decx::cuda_stream* S[3], const int flag)
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R16 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim->x / 2;

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 8 + bounded_kernel_R16,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    size_t offset_lin = 0, offset_ker = 0;
    // strat the main loop
    MIF<void> rep_const_mem0, rep_const_mem1;
    decx::utils::set_mutex_memory_state<void>(&rep_const_mem0, &rep_const_mem1);
    const size_t CM_offset = kernel->width * kernel->height * sizeof(de::Half);
    size_t sym_cpy_offset = 0;

    main_loop_regulable_R_mk(
        hconv2_kernel_within16x16_offset(Dmem1->mem, Dmem3->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        hconv2_kernel_within16x16_offset(Dmem2->mem, Dmem4->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        conv3_main_loop_MemCpyHtoD_BC_MK(bounded_kernel_R16, bounded_kernel_R16, de::Half, 8),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_hconv2_mk_exact16x16_BC(int2 * Dsrc_alloc_dim, int2 * Ddst_alloc_dim,
        int2 * ker_dim,
        decx::_MatrixArray*src, decx::_MatrixArray*kernel, decx::_MatrixArray*dst,
        MIF<float4>*Dmem1, MIF<float4>*Dmem2, MIF<float4>*Dmem3, MIF<float4>*Dmem4,
    decx::cuda_stream* S[3], const int flag)
{
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 8 + bounded_kernel_R16,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    size_t offset_lin = 0, offset_ker = 0;
    // strat the main loop
    MIF<void> rep_const_mem0, rep_const_mem1;
    decx::utils::set_mutex_memory_state<void>(&rep_const_mem0, &rep_const_mem1);
    const size_t CM_offset = kernel->width * kernel->height * sizeof(de::Half);
    size_t sym_cpy_offset = 0;

    main_loop_regulable_R_mk(
        hconv2_kernel_exact16x16_offset(Dmem1->mem, Dmem3->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        hconv2_kernel_exact16x16_offset(Dmem2->mem, Dmem4->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, sym_cpy_offset / sizeof(de::Half), S[0], flag),
        conv3_main_loop_MemCpyHtoD_BC_MK(bounded_kernel_R16, bounded_kernel_R16, de::Half, 8),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




// **************************************************************************************************************************



namespace decx
{
    static void _Conv2_BC_R8x8_MK_fp16(
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle, const int flag);


    static void _Conv2_BC_R8x16_MK_fp16(
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle, const int flag);


    static void _Conv2_BC_R16x8_MK_fp16(
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle, const int flag);


    static void _Conv2_BC_R16x16_MK_fp16(
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle, const int flag);


    static void hConv2_border_zero_mk(
        decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle, const int flag);
}


// single kernel
static void decx::_Conv2_BC_R8x8_MK_fp16(
    decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH *handle, const int flag)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 128) * bounded_kernel_R8 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R8 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R8 / 4;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R8 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::cuda_stream* S[3];
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    }

    decx::PtrInfo<float4> r_Dmem1, r_Dmem2, r_Dmem3, r_Dmem4;
    if (decx::alloc::_device_malloc(&r_Dmem1, dev_src_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&r_Dmem2, dev_dst_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&r_Dmem3, dev_src_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&r_Dmem4, dev_dst_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }

    Dmem1.mem = r_Dmem1.ptr;            Dmem2.mem = r_Dmem2.ptr;
    Dmem3.mem = r_Dmem3.ptr;            Dmem4.mem = r_Dmem4.ptr;

    uint offset_lin = 0, offset_ker = 0;

    for (int k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(decx::Const_Mem, (de::Half*)kernel->MatptrArr.ptr[0] + offset_ker,
            kernel->width * sizeof(de::Half), offset_lin * sizeof(de::Half), cudaMemcpyHostToDevice, S[0]->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        decx::main_loop_hconv2_mk_exact8x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S, flag);
    }
    else {
        decx::main_loop_hconv2_mk_within8x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S, flag);
    }

    checkCudaErrors(cudaFree(Dmem1.mem));
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }

    decx::alloc::_device_dealloc(&r_Dmem1);
    decx::alloc::_device_dealloc(&r_Dmem2);
    decx::alloc::_device_dealloc(&r_Dmem3);
    decx::alloc::_device_dealloc(&r_Dmem4);
}




// single kernel
static void decx::_Conv2_BC_R8x16_MK_fp16(
    decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle, const int flag)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 128) * bounded_kernel_R16 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R8 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 4;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R8 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::cuda_stream* S[3];
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    }

    decx::PtrInfo<float4> r_Dmem1, r_Dmem2, r_Dmem3, r_Dmem4;
    if (decx::alloc::_device_malloc(&r_Dmem1, dev_src_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&r_Dmem2, dev_dst_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&r_Dmem3, dev_src_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&r_Dmem4, dev_dst_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }

    Dmem1.mem = r_Dmem1.ptr;            Dmem2.mem = r_Dmem2.ptr;
    Dmem3.mem = r_Dmem3.ptr;            Dmem4.mem = r_Dmem4.ptr;

    uint offset_lin = 0, offset_ker = 0;

    for (int k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(decx::Const_Mem, (de::Half*)kernel->MatptrArr.ptr[0] + offset_ker,
            kernel->width * sizeof(de::Half), offset_lin * sizeof(de::Half), cudaMemcpyHostToDevice, S[0]->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        decx::main_loop_hconv2_mk_exact8x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S, flag);
    }
    else {
        decx::main_loop_hconv2_mk_within8x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S, flag);
    }

    checkCudaErrors(cudaFree(Dmem1.mem));
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }

    decx::alloc::_device_dealloc(&r_Dmem1);
    decx::alloc::_device_dealloc(&r_Dmem2);
    decx::alloc::_device_dealloc(&r_Dmem3);
    decx::alloc::_device_dealloc(&r_Dmem4);
}




// single kernel
static void decx::_Conv2_BC_R16x8_MK_fp16(
    decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle, const int flag)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 128) * bounded_kernel_R8 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R16 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R8 / 4;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::cuda_stream* S[3];
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    }

    decx::PtrInfo<float4> r_Dmem1, r_Dmem2, r_Dmem3, r_Dmem4;
    if (decx::alloc::_device_malloc(&r_Dmem1, dev_src_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&r_Dmem2, dev_dst_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&r_Dmem3, dev_src_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&r_Dmem4, dev_dst_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }

    Dmem1.mem = r_Dmem1.ptr;            Dmem2.mem = r_Dmem2.ptr;
    Dmem3.mem = r_Dmem3.ptr;            Dmem4.mem = r_Dmem4.ptr;

    uint offset_lin = 0, offset_ker = 0;

    for (int k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(decx::Const_Mem, (de::Half*)kernel->MatptrArr.ptr[0] + offset_ker,
            kernel->width * sizeof(de::Half), offset_lin * sizeof(de::Half), cudaMemcpyHostToDevice, S[0]->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {
        decx::main_loop_hconv2_mk_exact16x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S, flag);
    }
    else {
        decx::main_loop_hconv2_mk_within16x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S, flag);
    }

    checkCudaErrors(cudaFree(Dmem1.mem));
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }

    decx::alloc::_device_dealloc(&r_Dmem1);
    decx::alloc::_device_dealloc(&r_Dmem2);
    decx::alloc::_device_dealloc(&r_Dmem3);
    decx::alloc::_device_dealloc(&r_Dmem4);
}





// single kernel
static void decx::_Conv2_BC_R16x16_MK_fp16(
    decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle, const int flag)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 128) * bounded_kernel_R16 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R16 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 4;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::cuda_stream* S[3];
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    }

    decx::PtrInfo<float4> r_Dmem1, r_Dmem2, r_Dmem3, r_Dmem4;
    if (decx::alloc::_device_malloc(&r_Dmem1, dev_src_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&r_Dmem2, dev_dst_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&r_Dmem3, dev_src_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&r_Dmem4, dev_dst_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::device_AllocateFailure(handle);
        return;
    }

    Dmem1.mem = r_Dmem1.ptr;            Dmem2.mem = r_Dmem2.ptr;
    Dmem3.mem = r_Dmem3.ptr;            Dmem4.mem = r_Dmem4.ptr;

    uint offset_lin = 0, offset_ker = 0;

    for (int k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(decx::Const_Mem, (de::Half*)kernel->MatptrArr.ptr[0] + offset_ker,
            kernel->width * sizeof(de::Half), offset_lin * sizeof(de::Half), cudaMemcpyHostToDevice, S[0]->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {
        decx::main_loop_hconv2_mk_exact16x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S, flag);
    }
    else {
        decx::main_loop_hconv2_mk_within16x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S, flag);
    }

    checkCudaErrors(cudaFree(Dmem1.mem));
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }

    decx::alloc::_device_dealloc(&r_Dmem1);
    decx::alloc::_device_dealloc(&r_Dmem2);
    decx::alloc::_device_dealloc(&r_Dmem3);
    decx::alloc::_device_dealloc(&r_Dmem4);
}



// ******************************************************************************************************************************

static void decx::hConv2_border_zero_mk(
    decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst, de::DH* handle, const int flag)
{
    int2 half_ker_dim;
    half_ker_dim.x = kernel->width / 2;                half_ker_dim.y = kernel->height / 2;

    dst->re_construct(src->type, src->width,
        src->height,
        src->ArrayNumber,
        decx::DATA_STORE_TYPE::Page_Locked);


    if (half_ker_dim.x < bounded_kernel_R8 + 1) {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::_Conv2_BC_R8x8_MK_fp16(src, kernel, dst, handle, flag);
        }
        else {
            decx::_Conv2_BC_R16x8_MK_fp16(src, kernel, dst, handle, flag);
        }
    }
    else {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::_Conv2_BC_R8x16_MK_fp16(src, kernel, dst, handle, flag);
        }
        else {
            decx::_Conv2_BC_R16x16_MK_fp16(src, kernel, dst, handle, flag);
        }
    }
}


#endif