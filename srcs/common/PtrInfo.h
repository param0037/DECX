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


#ifndef _PTR_INFO_H_
#define _PTR_INFO_H_

#include <include.h>
#include <log_console.h>
#include <decx_alloc_interface.h>


namespace decx
{
    template <typename _Ty>
    class PtrInfo;


    template <typename _Ty>
    class Ptr2D_Info;
}


template <typename _Ty>
class decx::PtrInfo
{
private:
    DecxMemoryHandler_t block;
    _Ty*                ptr;
    DecxMemoryType_e    _mem_type;

public:
    PtrInfo() {
        this->block = NULL;
        this->ptr = NULL;
        this->_mem_type = DecxMemoryType_e::PAGABLE;
    }

    template <typename _Type_dst>
    decx::PtrInfo<_Type_dst> _type_cast()
    {
        decx::PtrInfo<_Type_dst> _dst;
        _dst.block = this->block;
        _dst.ptr = reinterpret_cast<_Type_dst*>(this->ptr);

        return _dst;
    }

    int32_t IsValid()
    {
        return this->ptr != NULL ? 1 : 0;
    }


    template<typename _type_out>
    explicit operator _type_out() { return (_type_out)this->ptr; }


    template<typename _type_out>
    explicit operator _type_out() const { return (_type_out)this->ptr; }


    template <typename _Out_Ptr = _Ty>
    _Out_Ptr* GetRawPtr()
    {
        return (_Out_Ptr*)this->ptr;
    }

    template <typename _Out_Ptr = _Ty>
    const _Out_Ptr* GetRawPtrConst() const
    {
        return (const _Out_Ptr*)this->ptr;
    }


#ifdef _DECX_CUDA_PARTS_
    int32_t Allocate(const uint64_t size, const DecxMemoryType_e alloc_type, de::DH* handle = nullptr, const bool zero_initialize = true, decx::cuda_stream* S = nullptr)
#else
    int32_t Allocate(const uint64_t size, const DecxMemoryType_e alloc_type, de::DH* handle = nullptr, const bool zero_initialize = true)
#endif
    {
        this->_mem_type = alloc_type;
        int32_t rval = 0;

        switch (alloc_type)
        {
        case PAGABLE:
            rval |= DecxAllocPagable(&this->block, size, (void**)(&this->ptr));
            if (zero_initialize){
                rval |= DecxMemset(this->block, size, 0);
            }
            if (handle != nullptr && rval != 0){
                decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
            }
            break;

#ifdef _DECX_CUDA_PARTS_
        case CUDA_DEVICE:
            rval |= DecxAllocCUDA(&this->block, size, (void**)(&this->ptr));
            if (zero_initialize){
                rval |= DecxCUDAMemset(this->block, size, 0, S);
            }
            if (handle != nullptr && rval != 0){
                decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION, DEV_ALLOC_FAIL);
            }
            break;
#endif

        default:
            return -1;
        }
        return rval;
    }


    #ifdef _DECX_CUDA_PARTS_
    int32_t Reallocate(const uint64_t size, de::DH* handle = nullptr, const bool zero_initialize = true, decx::cuda_stream* S = nullptr, const bool lazy_alloc = false)
#else
    int32_t Reallocate(const uint64_t size, de::DH* handle = nullptr, const bool zero_initialize = true, const bool lazy_alloc = false)
#endif
    {
        int32_t rval = 0;

        switch (this->_mem_type)
        {
        case DecxMemoryType_e::PAGABLE:
            if (lazy_alloc)
                rval |= DecxReallocPagableLazy(&this->block, size, (void**)(&this->ptr));
            else
                rval |= DecxReallocPagable(&this->block, size, (void**)(&this->ptr));
            if (zero_initialize){
                rval |= DecxMemset(this->block, size, 0);
            }
            if (handle != nullptr && rval != 0){
                decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
            }
        return rval;

#ifdef _DECX_CUDA_PARTS_
        case DecxMemoryType_e::CUDA_DEVICE:
            if (lazy_alloc)
                rval |= DecxReallocCUDALazy(&this->block, size, (void**)(&this->ptr));
            else
                rval |= DecxReallocCUDA(&this->block, size, (void**)(&this->ptr));
            if (zero_initialize){
                rval |= DecxCUDAMemset(this->block, size, 0, S);
            }
            if (handle != nullptr && rval != 0){
                decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION, DEV_ALLOC_FAIL);
            }
        return rval;
#endif

        default:
            return -1;
        }
    }


    int32_t Free()
    {
        switch (this->_mem_type)
        {
        case PAGABLE:
            return DecxFreePagable(this->block);

#ifdef _DECX_CUDA_PARTS_
        case CUDA_DEVICE:
            return DecxFreeCUDA(this->block);
#endif
        
        default:
            break;
        }
        return 0;
    }


    template <typename _type_out = _Ty>
    _type_out* operator+(const uint64_t offset)
    {
        return this->GetRawPtr<_type_out>() + offset;
    }


    _Ty& operator[](const uint64_t idx)
    {
        return *(this->GetRawPtr<_Ty>() + idx);
    }


    const _Ty& operator[](const uint64_t idx) const
    {
        return *(this->GetRawPtrConst<_Ty>() + idx);
    }
    

    int32_t AllocateRef()
    {
        switch (this->_mem_type)
        {
        case PAGABLE:
        case PAGELOCKED:
            return DecxAllocPagableRef(this->block, (void**)(&this->ptr));

#ifdef _DECX_CUDA_PARTS_
        case CUDA_DEVICE:
            return DecxAllocCUDARef(this->block, (void**)(&this->ptr));
#endif
        default:
            return -1;
        }
    }
};



template <typename _Ty>
class decx::Ptr2D_Info
{
private:
    decx::PtrInfo<_Ty> _ptr;
    uint2 _dims;

public:
    Ptr2D_Info() {
        this->_dims = make_uint2(0, 0);
    }

    Ptr2D_Info(decx::PtrInfo<_Ty> _ptr_info, const uint2 dims)
    {
        this->_dims = dims;
        this->_ptr = _ptr_info;
    }

    int32_t IsValid()
    {
        return this->_ptr.IsValid();
    }

    template <typename _Out_Ptr = _Ty>
    _Out_Ptr* GetRawPtr()
    {
        return this->_ptr.template GetRawPtr<_Out_Ptr>();
    }

    void SetDims(const uint2 dims) {this->_dims = dims;}
    void SetDims(const uint32_t x, const uint32_t y) {this->_dims.x = x; this->_dims.y = y; }


    const uint2& getDims() const {return this->_dims; }
    

    template <typename _Out_Ptr = _Ty>
    _Out_Ptr* GetRawPtrConst() const
    {
        return this->_ptr.template GetRawPtrConst<_Out_Ptr>();
    }


#ifdef _DECX_CUDA_PARTS_
    int32_t Allocate(const DecxMemoryType_e mem_type,   const uint32_t element_size = sizeof(_Ty), 
                     de::DH* handle = nullptr,          const bool zero_initialize = true,
                    decx::cuda_stream* S = nullptr)
#else
    int32_t Allocate(const DecxMemoryType_e mem_type,   const uint32_t element_size = sizeof(_Ty), 
                     de::DH* handle = nullptr,          const bool zero_initialize = true)
#endif
    {
        const uint64_t size_alloca = (uint64_t)this->_dims.x * (uint64_t)this->_dims.y * element_size;
#ifdef _DECX_CUDA_PARTS_
        int32_t rval = this->_ptr.Allocate(size_alloca, mem_type, handle, zero_initialize, S);
#else
        int32_t rval = this->_ptr.Allocate(size_alloca, mem_type, handle, zero_initialize);
#endif
        return rval;
    }

    
#ifdef _DECX_CUDA_PARTS_
    int32_t Reallocate(const uint32_t element_size = sizeof(_Ty),   de::DH* handle = nullptr,
                    const bool zero_initialize = true,           decx::cuda_stream* S = nullptr, const bool lazy_alloc = false)
#else
    int32_t Reallocate(const uint32_t element_size = sizeof(_Ty),   de::DH* handle = nullptr,
                    const bool zero_initialize = true,              const bool lazy_alloc = false)
#endif
    {
        const uint64_t size_alloca = (uint64_t)this->_dims.x * (uint64_t)this->_dims.y * element_size;
#ifdef _DECX_CUDA_PARTS_
        int32_t rval = this->_ptr.Reallocate(size_alloca, handle, zero_initialize, S, lazy_alloc);
#else
        int32_t rval = this->_ptr.Reallocate(size_alloca, handle, zero_initialize, lazy_alloc);
#endif
        return rval;
    }


    int32_t AllocateRef()
    {
        return this->_ptr.AllocateRef();
    }


    int32_t Free()
    {
        return this->_ptr.Free();
    }


    template <typename _type_out = _Ty>
    _type_out* operator+(const uint64_t offset)
    {
        return this->GetRawPtr<_type_out>() + offset;
    }

    
    _Ty& operator[](const uint64_t idx)
    {
        return *(this->GetRawPtr<_Ty>() + idx);
    }

    
    const _Ty& operator[](const uint64_t idx) const
    {
        return *(this->GetRawPtrConst<_Ty>() + idx);
    }
};



#endif