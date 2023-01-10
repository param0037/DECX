/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _GEMM3_FP32_H_
#define _GEMM3_FP32_H_

#include "../../../../classes/MatrixArray.h"
#include "GEMM_fp32_kernel_callers.h"


namespace decx
{
    static void sGEMM3_caller(decx::_MatrixArray* _A, decx::_MatrixArray* _B, decx::_MatrixArray* _dst,
        decx::cuda_stream** S, de::DH* handle);


    static void sGEMM3_ABC_caller(decx::_MatrixArray* _A, decx::_MatrixArray* _B, decx::_MatrixArray* _C,
        decx::_MatrixArray* _dst, decx::cuda_stream** S, de::DH* handle);
}


static void decx::sGEMM3_caller(decx::_MatrixArray* _A, decx::_MatrixArray* _B, decx::_MatrixArray* _dst, 
    decx::cuda_stream** S, de::DH *handle)
{
    uint pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A->width % 16) != 0) ? (decx::utils::ceil<uint>(_A->width, 16) * 16) : _A->width;
    pitch_B = ((_B->width % 128) != 0) ? (decx::utils::ceil<uint>(_B->width, 128) * 128) : _B->width;
    hA = ((_A->height % 128) != 0) ? (decx::utils::ceil<uint>(_A->height, 128) * 128) : _A->height;

    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    /*
    * [0, 3] -> A;
    * [1, 4] -> B;
    * [2, 5] -> dst;
    */
    decx::PtrInfo<float> dev_tmp[6];            // this is the total buffer that this function requires
    if (decx::alloc::_device_malloc(&dev_tmp[0], mem_A * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_tmp[1], mem_B * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_tmp[2], mem_dst * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_tmp[3], mem_A * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_tmp[4], mem_B * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_tmp[5], mem_dst * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    decx::alloc::MIF<float> d_mem[6];
    d_mem[0].mem = dev_tmp[0].ptr;                  d_mem[1].mem = dev_tmp[1].ptr;
    d_mem[2].mem = dev_tmp[2].ptr;                  d_mem[3].mem = dev_tmp[3].ptr;
    d_mem[4].mem = dev_tmp[4].ptr;                  d_mem[5].mem = dev_tmp[5].ptr;

    checkCudaErrors(cudaMemcpy2DAsync(d_mem[0].mem,
        pitch_A * sizeof(float), _A->MatptrArr.ptr[0], _A->pitch * sizeof(float),
        _A->width * sizeof(float), _A->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

    checkCudaErrors(cudaMemcpy2DAsync(d_mem[1].mem,
        pitch_B * sizeof(float), _B->MatptrArr.ptr[0], _B->pitch * sizeof(float),
        _B->width * sizeof(float), _B->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

    decx::utils::set_mutex_memory_state(&d_mem[0], &d_mem[3]);
    decx::utils::set_mutex_memory_state(&d_mem[1], &d_mem[4]);

    checkCudaErrors(cudaStreamSynchronize(S[1]->get_raw_stream_ref()));

    for (int i = 0; i < _A->ArrayNumber; ++i)
    {
        if (i > 0) {
            if (d_mem[2].leading) {
                checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[i - 1],
                    _dst->pitch * sizeof(float), d_mem[2].mem, pitch_B * sizeof(float), _dst->width * sizeof(float),
                    _dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
                d_mem[2]._using = true;
            }
            else {
                checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[i - 1],
                    _dst->pitch * sizeof(float), d_mem[5].mem, pitch_B * sizeof(float), _dst->width * sizeof(float),
                    _dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
                d_mem[5]._using = true;
            }
        }

        if (d_mem[0].leading) {
            sGEMM_part(d_mem[0].mem, d_mem[1].mem, d_mem[2].mem, pitch_A, pitch_B, hA, S[0]->get_raw_stream_ptr());
            
            decx::utils::set_mutex_memory3_using(&d_mem[0], &d_mem[1], &d_mem[2]);
            decx::utils::set_mutex_memory_state(&d_mem[2], &d_mem[5]);
        }
        else {
            sGEMM_part(d_mem[3].mem, d_mem[4].mem, d_mem[5].mem, pitch_A, pitch_B, hA, S[0]->get_raw_stream_ptr());
            
            decx::utils::set_mutex_memory3_using(&d_mem[3], &d_mem[4], &d_mem[5]);
            decx::utils::set_mutex_memory_state(&d_mem[5], &d_mem[2]);
        }

        if (i < _A->ArrayNumber - 1) {
            if (!d_mem[0]._using) {
                checkCudaErrors(cudaMemcpy2DAsync(d_mem[0].mem,
                    pitch_A * sizeof(float), _A->MatptrArr.ptr[i + 1], _A->pitch * sizeof(float),
                    _A->width * sizeof(float), _A->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

                checkCudaErrors(cudaMemcpy2DAsync(d_mem[1].mem,
                    pitch_B * sizeof(float), _B->MatptrArr.ptr[i + 1], _B->pitch * sizeof(float),
                    _B->width * sizeof(float), _B->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

                decx::utils::set_mutex_memory_state(&d_mem[0], &d_mem[3]);
                decx::utils::set_mutex_memory_state(&d_mem[1], &d_mem[4]);
            }
            else {
                checkCudaErrors(cudaMemcpy2DAsync(d_mem[3].mem,
                    pitch_A * sizeof(float), _A->MatptrArr.ptr[i + 1], _A->pitch * sizeof(float),
                    _A->width * sizeof(float), _A->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

                checkCudaErrors(cudaMemcpy2DAsync(d_mem[4].mem,
                    pitch_B * sizeof(float), _B->MatptrArr.ptr[i + 1], _B->pitch * sizeof(float),
                    _B->width * sizeof(float), _B->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

                decx::utils::set_mutex_memory_state(&d_mem[3], &d_mem[0]);
                decx::utils::set_mutex_memory_state(&d_mem[4], &d_mem[1]);
            }
        }
        checkCudaErrors(cudaDeviceSynchronize());

        decx::utils::set_mutex_memory3_idle(&d_mem[0], &d_mem[1], &d_mem[2]);
        decx::utils::set_mutex_memory3_idle(&d_mem[3], &d_mem[4], &d_mem[5]);
    }

    if (d_mem[2].leading) {
        checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[_A->ArrayNumber - 1],
            _dst->pitch * sizeof(float), d_mem[2].mem, pitch_B * sizeof(float), _dst->width * sizeof(float),
            _dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));                                
    }                                                                                                            
    else {
        checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[_A->ArrayNumber - 1],
            _dst->pitch * sizeof(float), d_mem[5].mem, pitch_B * sizeof(float), _dst->width * sizeof(float),
            _dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));                                 
    }
    //checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamSynchronize(S[2]->get_raw_stream_ref()));

#pragma unroll 6
    for (int i = 0; i < 6; ++i) {
        decx::alloc::_device_dealloc(&dev_tmp[i]);
    }
}



static void decx::sGEMM3_ABC_caller(decx::_MatrixArray* _A, decx::_MatrixArray* _B, decx::_MatrixArray* _C,
    decx::_MatrixArray* _dst, decx::cuda_stream** S, de::DH *handle)
{
    uint pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A->width % 16) != 0) ? (decx::utils::ceil<uint>(_A->width, 16) * 16) : _A->width;
    pitch_B = ((_B->width % 128) != 0) ? (decx::utils::ceil<uint>(_B->width, 128) * 128) : _B->width;
    hA = ((_A->height % 128) != 0) ? (decx::utils::ceil<uint>(_A->height, 128) * 128) : _A->height;

    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    /*
    * [0, 3] -> A;
    * [1, 4] -> B;
    * [2, 5] -> dst;
    */
    decx::PtrInfo<float> dev_tmp[6];            // this is the total buffer that this function requires
    if (decx::alloc::_device_malloc(&dev_tmp[0], mem_A * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_tmp[1], mem_B * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_tmp[2], mem_dst * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_tmp[3], mem_A * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_tmp[4], mem_B * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dev_tmp[5], mem_dst * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    decx::alloc::MIF<float> d_mem[6];
    d_mem[0].mem = dev_tmp[0].ptr;                  d_mem[1].mem = dev_tmp[1].ptr;
    d_mem[2].mem = dev_tmp[2].ptr;                  d_mem[3].mem = dev_tmp[3].ptr;
    d_mem[4].mem = dev_tmp[4].ptr;                  d_mem[5].mem = dev_tmp[5].ptr;

    checkCudaErrors(cudaMemcpy2DAsync(d_mem[0].mem,
        pitch_A * sizeof(float), _A->MatptrArr.ptr[0], _A->pitch * sizeof(float),
        _A->width * sizeof(float), _A->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

    checkCudaErrors(cudaMemcpy2DAsync(d_mem[1].mem,
        pitch_B * sizeof(float), _B->MatptrArr.ptr[0], _B->pitch * sizeof(float),
        _B->width * sizeof(float), _B->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

    checkCudaErrors(cudaMemcpy2DAsync(d_mem[2].mem,
        pitch_B * sizeof(float), _C->MatptrArr.ptr[0], _C->pitch * sizeof(float),
        _C->width * sizeof(float), _C->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

    decx::utils::set_mutex_memory_state(&d_mem[0], &d_mem[3]);
    decx::utils::set_mutex_memory_state(&d_mem[1], &d_mem[4]);

    checkCudaErrors(cudaStreamSynchronize(S[1]->get_raw_stream_ref()));

    for (int i = 0; i < _A->ArrayNumber; ++i)
    {
        if (i > 0) {
            if (d_mem[2].leading) {
                checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[i - 1],
                    _dst->pitch * sizeof(float), d_mem[2].mem, pitch_B * sizeof(float), _dst->width * sizeof(float),
                    _dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
                d_mem[2]._using = true;
            }
            else {
                checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[i - 1],
                    _dst->pitch * sizeof(float), d_mem[5].mem, pitch_B * sizeof(float), _dst->width * sizeof(float),
                    _dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
                d_mem[5]._using = true;
            }
        }

        if (d_mem[0].leading) {
            sGEMM_part_ABC(d_mem[0].mem, d_mem[1].mem, d_mem[2].mem, d_mem[2].mem, pitch_A, pitch_B, hA, S[0]->get_raw_stream_ptr());

            decx::utils::set_mutex_memory3_using(&d_mem[0], &d_mem[1], &d_mem[2]);
            decx::utils::set_mutex_memory_state(&d_mem[2], &d_mem[5]);
        }
        else {
            sGEMM_part_ABC(d_mem[3].mem, d_mem[4].mem, d_mem[5].mem, d_mem[5].mem, pitch_A, pitch_B, hA, S[0]->get_raw_stream_ptr());

            decx::utils::set_mutex_memory3_using(&d_mem[3], &d_mem[4], &d_mem[5]);
            decx::utils::set_mutex_memory_state(&d_mem[5], &d_mem[2]);
        }

        if (i < _A->ArrayNumber - 1) {
            if (!d_mem[0]._using) {
                checkCudaErrors(cudaMemcpy2DAsync(d_mem[0].mem,
                    pitch_A * sizeof(float), _A->MatptrArr.ptr[i + 1], _A->pitch * sizeof(float),
                    _A->width * sizeof(float), _A->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

                checkCudaErrors(cudaMemcpy2DAsync(d_mem[1].mem,
                    pitch_B * sizeof(float), _B->MatptrArr.ptr[i + 1], _B->pitch * sizeof(float),
                    _B->width * sizeof(float), _B->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

                checkCudaErrors(cudaMemcpy2DAsync(d_mem[2].mem,
                    pitch_B * sizeof(float), _C->MatptrArr.ptr[i + 1], _C->pitch * sizeof(float),
                    _C->width * sizeof(float), _C->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

                decx::utils::set_mutex_memory_state(&d_mem[0], &d_mem[3]);
                decx::utils::set_mutex_memory_state(&d_mem[1], &d_mem[4]);
            }
            else {
                checkCudaErrors(cudaMemcpy2DAsync(d_mem[3].mem,
                    pitch_A * sizeof(float), _A->MatptrArr.ptr[i + 1], _A->pitch * sizeof(float),
                    _A->width * sizeof(float), _A->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

                checkCudaErrors(cudaMemcpy2DAsync(d_mem[4].mem,
                    pitch_B * sizeof(float), _B->MatptrArr.ptr[i + 1], _B->pitch * sizeof(float),
                    _B->width * sizeof(float), _B->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

                checkCudaErrors(cudaMemcpy2DAsync(d_mem[5].mem,
                    pitch_B * sizeof(float), _C->MatptrArr.ptr[i + 1], _C->pitch * sizeof(float),
                    _C->width * sizeof(float), _C->height, cudaMemcpyHostToDevice, S[1]->get_raw_stream_ref()));

                decx::utils::set_mutex_memory_state(&d_mem[3], &d_mem[0]);
                decx::utils::set_mutex_memory_state(&d_mem[4], &d_mem[1]);
            }
        }

        checkCudaErrors(cudaDeviceSynchronize());

        decx::utils::set_mutex_memory3_idle(&d_mem[0], &d_mem[1], &d_mem[2]);
        decx::utils::set_mutex_memory3_idle(&d_mem[3], &d_mem[4], &d_mem[5]);
    }

    if (d_mem[2].leading) {
        checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[_A->ArrayNumber - 1],
            _dst->pitch * sizeof(float), d_mem[2].mem, pitch_B * sizeof(float), _dst->width * sizeof(float),
            _dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[_A->ArrayNumber - 1],
            _dst->pitch * sizeof(float), d_mem[5].mem, pitch_B * sizeof(float), _dst->width * sizeof(float),
            _dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
    }

    checkCudaErrors(cudaStreamSynchronize(S[2]->get_raw_stream_ref()));

#pragma unroll 6
    for (int i = 0; i < 6; ++i) {
        decx::alloc::_device_dealloc(&dev_tmp[i]);
    }
}



#endif