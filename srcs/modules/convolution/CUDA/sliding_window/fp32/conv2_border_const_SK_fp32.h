/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.9.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.9.16
*/

#ifndef _CONV2_BORDER_CONST_SK_FP32_H_
#define _CONV2_BORDER_CONST_SK_FP32_H_


#include "../../../../core/basic.h"
#include "../Conv2_MC_macros.h"


using decx::_Matrix;
using decx::alloc::MIF;
using decx::_MatrixArray;


namespace decx
{
    static void main_loop_sconv2_sk_within8x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);



    static void main_loop_sconv2_sk_exact8x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);



    static void main_loop_sconv2_sk_within8x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);


    static void main_loop_sconv2_sk_exact8x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);


    static void main_loop_sconv2_sk_within16x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);


    static void main_loop_sconv2_sk_exact16x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);


    static void main_loop_sconv2_sk_within16x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);


    static void main_loop_sconv2_sk_exact16x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);


    // single kernel
    static void _Conv2_BC_R8x8_SK(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst);


    // single kernel
    static void _Conv2_BC_R8x16_SK(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst);


    // single kernel
    static void _Conv2_BC_R16x8_SK(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst);


    // single kernel
    static void _Conv2_BC_R16x16_SK(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst);


    static void sConv2_border_zero_sk(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst, de::DH* handle);
}






static void decx::main_loop_sconv2_sk_within8x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
    decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst,
    MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
    decx::cuda_stream* S[3])
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R8 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim->x / 2;

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 4 + bounded_kernel_R8,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(float),
        src->width * sizeof(float),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        sconv2_kernel_within8x8(Dmem1->mem, Dmem3->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        sconv2_kernel_within8x8(Dmem2->mem, Dmem4->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        conv3_main_loop_MemCpyHtoD_BC(bounded_kernel_R8, bounded_kernel_R8, float, 4),
        float)

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_sconv2_sk_exact8x8_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
    decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst,
    MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
    decx::cuda_stream* S[3])
{
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 4 + bounded_kernel_R8,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(float),
        src->width * sizeof(float),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        sconv2_kernel_exact8x8(Dmem1->mem, Dmem3->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        sconv2_kernel_exact8x8(Dmem2->mem, Dmem4->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        conv3_main_loop_MemCpyHtoD_BC(bounded_kernel_R8, bounded_kernel_R8, float, 4),
        float);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}





static void decx::main_loop_sconv2_sk_within8x16_BC(int2 * Dsrc_alloc_dim, int2 * Ddst_alloc_dim, int2 * ker_dim,
    decx::_MatrixArray*src, decx::_Matrix*kernel, decx::_MatrixArray*dst,
    MIF<float4>*Dmem1, MIF<float4>*Dmem2, MIF<float4>*Dmem3, MIF<float4>*Dmem4,
    decx::cuda_stream* S[3])
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R8 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim->x / 2;

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 4 + bounded_kernel_R16,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(float),
        src->width * sizeof(float),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        sconv2_kernel_within8x16(Dmem1->mem, Dmem3->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        sconv2_kernel_within8x16(Dmem2->mem, Dmem4->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        conv3_main_loop_MemCpyHtoD_BC(bounded_kernel_R8, bounded_kernel_R16, float, 4),
        float);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_sconv2_sk_exact8x16_BC(int2 * Dsrc_alloc_dim, int2 * Ddst_alloc_dim, int2 * ker_dim,
    decx::_MatrixArray*src, decx::_Matrix*kernel, decx::_MatrixArray*dst,
    MIF<float4>*Dmem1, MIF<float4>*Dmem2, MIF<float4>*Dmem3, MIF<float4>*Dmem4,
    decx::cuda_stream* S[3])
{
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R8 * 4 + bounded_kernel_R16,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(float),
        src->width * sizeof(float),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        sconv2_kernel_exact8x16(Dmem1->mem, Dmem3->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        sconv2_kernel_exact8x16(Dmem2->mem, Dmem4->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        conv3_main_loop_MemCpyHtoD_BC(bounded_kernel_R8, bounded_kernel_R16, float, 4),
        float);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_sconv2_sk_within16x8_BC(int2 * Dsrc_alloc_dim, int2 * Ddst_alloc_dim, int2 * ker_dim,
    decx::_MatrixArray*src, decx::_Matrix*kernel, decx::_MatrixArray*dst,
    MIF<float4>*Dmem1, MIF<float4>*Dmem2, MIF<float4>*Dmem3, MIF<float4>*Dmem4,
    decx::cuda_stream* S[3])
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R16 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim->x / 2;

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 4 + bounded_kernel_R8,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(float),
        src->width * sizeof(float),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        sconv2_kernel_within16x8(Dmem1->mem, Dmem3->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        sconv2_kernel_within16x8(Dmem2->mem, Dmem4->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        conv3_main_loop_MemCpyHtoD_BC(bounded_kernel_R16, bounded_kernel_R8, float, 4),
        float);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_sconv2_sk_exact16x8_BC(int2 * Dsrc_alloc_dim, int2 * Ddst_alloc_dim, int2 * ker_dim,
    decx::_MatrixArray*src, decx::_Matrix*kernel, decx::_MatrixArray*dst,
    MIF<float4>*Dmem1, MIF<float4>*Dmem2, MIF<float4>*Dmem3, MIF<float4>*Dmem4,
    decx::cuda_stream* S[3])
{
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 4 + bounded_kernel_R8,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(float),
        src->width * sizeof(float),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        sconv2_kernel_exact16x8(Dmem1->mem, Dmem3->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        sconv2_kernel_exact16x8(Dmem2->mem, Dmem4->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        conv3_main_loop_MemCpyHtoD_BC(bounded_kernel_R16, bounded_kernel_R8, float, 4),
        float);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
    Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}





static void decx::main_loop_sconv2_sk_within16x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
    decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst,
    MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
    decx::cuda_stream* S[3])
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R16 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim->x / 2;

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 4 + bounded_kernel_R16,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(float),
        src->width * sizeof(float),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        sconv2_kernel_within16x16(Dmem1->mem, Dmem3->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        sconv2_kernel_within16x16(Dmem2->mem, Dmem4->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        conv3_main_loop_MemCpyHtoD_BC(bounded_kernel_R16, bounded_kernel_R16, float, 4),
        float);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_sconv2_sk_exact16x16_BC(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
    decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst,
    MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
    decx::cuda_stream* S[3])
{
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(Dmem1->mem) + Dsrc_alloc_dim->x * bounded_kernel_R16 * 4 + bounded_kernel_R16,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(float),
        src->width * sizeof(float),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        sconv2_kernel_exact16x16(Dmem1->mem, Dmem3->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        sconv2_kernel_exact16x16(Dmem2->mem, Dmem4->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]),
        conv3_main_loop_MemCpyHtoD_BC(bounded_kernel_R16, bounded_kernel_R16, float, 4),
        float);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(float), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(float),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}



// **************************************************************************************************************************


// single kernel
static void decx::_Conv2_BC_R8x8_SK(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * bounded_kernel_R8 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R8 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R8 / 2;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R8 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::PtrInfo<float4> dev_src1, dev_src2, dev_dst1, dev_dst2;
    if (decx::alloc::_device_malloc(&dev_src1, dev_src_size * sizeof(float4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_src2, dev_src_size * sizeof(float4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_dst1, dev_dst_size * sizeof(float4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_dst2, dev_dst_size * sizeof(float4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    Dmem1.mem = dev_src1.ptr;               Dmem2.mem = dev_src2.ptr;
    Dmem3.mem = dev_dst1.ptr;               Dmem4.mem = dev_dst2.ptr;

    decx::cuda_stream* S[3];
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    }

    uint offset_lin = 0, offset_ker = 0;

    for (uint k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(decx::Const_Mem, (float*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(float), offset_lin * sizeof(float), cudaMemcpyHostToDevice, S[0]->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        main_loop_sconv2_sk_exact8x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }
    else {
        main_loop_sconv2_sk_within8x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }

    decx::alloc::_device_dealloc(&dev_src1);
    decx::alloc::_device_dealloc(&dev_src2);
    decx::alloc::_device_dealloc(&dev_dst1);
    decx::alloc::_device_dealloc(&dev_dst2);

#pragma unroll
    for (uint i = 0; i < 3; ++i) {
        S[i]->detach();
    }
}




// single kernel
static void decx::_Conv2_BC_R8x16_SK(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * bounded_kernel_R16 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R8 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 2;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R8 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::PtrInfo<float4> dev_src1, dev_src2, dev_dst1, dev_dst2;
    if (decx::alloc::_device_malloc(&dev_src1, dev_src_size * sizeof(float4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_src2, dev_src_size * sizeof(float4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_dst1, dev_dst_size * sizeof(float4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_dst2, dev_dst_size * sizeof(float4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    Dmem1.mem = dev_src1.ptr;               Dmem2.mem = dev_src2.ptr;
    Dmem3.mem = dev_dst1.ptr;               Dmem4.mem = dev_dst2.ptr;

    decx::cuda_stream* S[3];
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    }

    uint offset_lin = 0, offset_ker = 0;

    for (uint k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(decx::Const_Mem, (float*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(float), offset_lin * sizeof(float), cudaMemcpyHostToDevice, S[0]->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        main_loop_sconv2_sk_exact8x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }
    else {
        main_loop_sconv2_sk_within8x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }

    decx::alloc::_device_dealloc(&dev_src1);
    decx::alloc::_device_dealloc(&dev_src2);
    decx::alloc::_device_dealloc(&dev_dst1);
    decx::alloc::_device_dealloc(&dev_dst2);

#pragma unroll
    for (uint i = 0; i < 3; ++i) {
        S[i]->detach();
    }
}



// single kernel
static void decx::_Conv2_BC_R16x8_SK(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * bounded_kernel_R8 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R16 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R8 / 2;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::cuda_stream* S[3];
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    }

    decx::PtrInfo<float4> dev_src1, dev_src2, dev_dst1, dev_dst2;
    if (decx::alloc::_device_malloc(&dev_src1, dev_src_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_src2, dev_src_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_dst1, dev_dst_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_dst2, dev_dst_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    Dmem1.mem = dev_src1.ptr;               Dmem2.mem = dev_src2.ptr;
    Dmem3.mem = dev_dst1.ptr;               Dmem4.mem = dev_dst2.ptr;

    uint offset_lin = 0, offset_ker = 0;

    for (int k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(decx::Const_Mem, (float*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(float), offset_lin * sizeof(float), cudaMemcpyHostToDevice, S[0]->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {
        main_loop_sconv2_sk_exact16x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }
    else {
        main_loop_sconv2_sk_within16x8_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }

    decx::alloc::_device_dealloc(&dev_src1);
    decx::alloc::_device_dealloc(&dev_src2);
    decx::alloc::_device_dealloc(&dev_dst1);
    decx::alloc::_device_dealloc(&dev_dst2);

#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }
}





// single kernel
static void decx::_Conv2_BC_R16x16_SK(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 64) * bounded_kernel_R16 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R16 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 2;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    decx::cuda_stream* S[3];
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    }

    decx::PtrInfo<float4> dev_src1, dev_src2, dev_dst1, dev_dst2;
    if (decx::alloc::_device_malloc(&dev_src1, dev_src_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_src2, dev_src_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_dst1, dev_dst_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_dst2, dev_dst_size * sizeof(float4), true, S[0])) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    Dmem1.mem = dev_src1.ptr;               Dmem2.mem = dev_src2.ptr;
    Dmem3.mem = dev_dst1.ptr;               Dmem4.mem = dev_dst2.ptr;

    uint offset_lin = 0, offset_ker = 0;

    for (int k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(decx::Const_Mem, (float*)kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(float), offset_lin * sizeof(float), cudaMemcpyHostToDevice, S[0]->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {
        main_loop_sconv2_sk_exact16x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }
    else {
        main_loop_sconv2_sk_within16x16_BC(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }

    decx::alloc::_device_dealloc(&dev_src1);
    decx::alloc::_device_dealloc(&dev_src2);
    decx::alloc::_device_dealloc(&dev_dst1);
    decx::alloc::_device_dealloc(&dev_dst2);

#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }
}



// ******************************************************************************************************************************

static void decx::sConv2_border_zero_sk(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst, de::DH* handle)
{
    int2 half_ker_dim;
    half_ker_dim.x = kernel->width / 2;                half_ker_dim.y = kernel->height / 2;

    dst->re_construct(src->type, src->width,
        src->height,
        src->ArrayNumber,
        decx::DATA_STORE_TYPE::Page_Locked);

    if (half_ker_dim.x < bounded_kernel_R8 + 1) {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::_Conv2_BC_R8x8_SK(src, kernel, dst);
        }
        else {
            decx::_Conv2_BC_R16x8_SK(src, kernel, dst);
        }
    }
    else {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::_Conv2_BC_R8x16_SK(src, kernel, dst);
        }
        else {
            decx::_Conv2_BC_R16x16_SK(src, kernel, dst);
        }
    }
}


#endif