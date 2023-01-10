/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/



#include "../../../classes/Vector.h"
#include "../../../classes/GPU_Vector.h"
#include "../Add_kernel.cuh"
#include "../../../core/basic.h"



namespace de
{
    namespace cuda
    {
        _DECX_API_  de::DH Add(de::GPU_Vector& A, de::GPU_Vector& B, de::GPU_Vector& dst);



        _DECX_API_  de::DH Add(de::GPU_Vector& src, void* __x, de::GPU_Vector& dst);
    }
}



de::DH de::cuda::Add(de::GPU_Vector& A, de::GPU_Vector& B, de::GPU_Vector& dst)
{
    decx::_GPU_Vector& _A = dynamic_cast<decx::_GPU_Vector&>(A);
    decx::_GPU_Vector& _B = dynamic_cast<decx::_GPU_Vector&>(B);
    decx::_GPU_Vector& _dst = dynamic_cast<decx::_GPU_Vector&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A._length != _B._length) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    const size_t len = (size_t)_A._length;
    switch (_A.type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::dev_Kadd_m((de::Half*)_A.Vec.ptr, (de::Half*)_B.Vec.ptr, (de::Half*)_dst.Vec.ptr, len);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::dev_Kadd_m((float*)_A.Vec.ptr, (float*)_B.Vec.ptr, (float*)_dst.Vec.ptr, len);
        break;

    case decx::_DATA_TYPES_FLAGS_::_INT32_:
        decx::calc::dev_Kadd_m((int*)_A.Vec.ptr, (int*)_B.Vec.ptr, (int*)_dst.Vec.ptr, len);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        decx::calc::dev_Kadd_m((double*)_A.Vec.ptr, (double*)_B.Vec.ptr, (double*)_dst.Vec.ptr, len);
        break;
    default:
        break;
    }

    decx::Success(&handle);
    return handle;
}




de::DH de::cuda::Add(de::GPU_Vector& A, void* __x, de::GPU_Vector& dst)
{
    decx::_GPU_Vector& _A = dynamic_cast<decx::_GPU_Vector&>(A);
    decx::_GPU_Vector& _dst = dynamic_cast<decx::_GPU_Vector&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.length != _dst.length) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    if (__x == NULL) {
        Print_Error_Message(4, "Can't find entity of parameter '__x'\n");
        decx::err::InvalidParam(&handle);
        return handle;
    }

    const size_t len = (size_t)_A._length;
    switch (_A.type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::dev_Kadd_c((de::Half*)_A.Vec.ptr, *((de::Half*)__x), (de::Half*)_dst.Vec.ptr, len);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::dev_Kadd_c((float*)_A.Vec.ptr, *((float*)__x), (float*)_dst.Vec.ptr, len);
        break;

    case decx::_DATA_TYPES_FLAGS_::_INT32_:
        decx::calc::dev_Kadd_c((int*)_A.Vec.ptr, *((int*)__x), (int*)_dst.Vec.ptr, len);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        decx::calc::dev_Kadd_c((double*)_A.Vec.ptr, *((double*)__x), (double*)_dst.Vec.ptr, len);
        break;
    default:
        break;
    }

    decx::Success(&handle);
    return handle;
}
