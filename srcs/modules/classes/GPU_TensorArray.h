///**
//*    ---------------------------------------------------------------------
//*    Author : Wayne Anderson
//*    Date   : 2021.04.16
//*    ---------------------------------------------------------------------
//*    This is a part of the open source program named "DECX", copyright c Wayne,
//*    2021.04.16
//*/
//
//
//#ifndef _GPU_TENSORARRAY_H_
//#define _GPU_TENSORARRAY_H_
//
//#include "../core/basic.h"
//#include "../core/allocators.h"
//#include "../classes/classes_util.h"
//#include "store_types.h"
//#include "TensorArray.h"
//
//
//namespace de
//{
//    template <typename T>
//    class _DECX_API_ GPU_TensorArray
//    {
//    public:
//        GPU_TensorArray() {}
//
//
//        virtual uint Width() = 0;
//
//
//        virtual uint Height() = 0;
//
//
//        virtual uint Depth() = 0;
//
//
//        virtual uint TensorNum() = 0;
//
//
//        virtual void Load_from_host(de::TensorArray<T>& src) = 0;
//
//
//        virtual void Load_to_host(de::TensorArray<T>& src) = 0;
//
//
//        virtual de::GPU_TensorArray<T>& operator=(de::GPU_TensorArray<T>& src) = 0;
//
//
//        virtual void release() = 0;
//    };
//}
//
//
//#ifndef _DECX_COMBINED_
//
///**
//* The data storage structure is shown below
//* tensor_id
//*            <-------------------- dp_x_w --------------------->
//*             <---------------- width -------------->
//*             <-dpitch->
//*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] T            T
//*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |            |
//*    0       [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |            |
//*            ...                                     ...         |    height  |    hpitch(2x)
//*            ...                                     ...         |            |
//*            ...                                     ...         |            |
//*       ___> [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] _            _
//*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]]
//*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]]
//*    1       [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]]
//*            ...                                     ...
//*            ...                                     ...
//*            ...                                     ...
//*       ___> [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]]
//*    .
//*    .
//*    .
//*
//* Where : the vector along depth-axis
//*    <------------ dpitch ----------->
//*    <---- pitch ------>
//*    [x x x x x x x x x 0 0 0 0 0 0 0]
//*/
//
//namespace decx
//{
//    template <typename T>
//    class _GPU_TensorArray : public de::GPU_TensorArray<T>
//    {
//    private:
//        void _attribute_assign(const uint _width, const uint _height, const uint _depth, const uint _tensor_num);
//
//
//        void alloc_data_space();
//
//
//        void re_alloc_data_space();
//
//    public:
//        uint width,
//            height,
//            depth,
//            tensor_num;
//
//        // The data pointer
//        decx::PtrInfo<T> TensArr;
//        // The pointer array for the pointers of each tensor in the TensorArray
//        decx::PtrInfo<T*> TensptrArr;
//
//        uint dpitch;            // NOT IN BYTES, the true depth (4x)
//        uint wpitch;            // NOT IN BYTES, the true width (2x)
//        size_t dp_x_wp;            // NOT IN BYTES, true depth multiply true width
//
//        /*
//         * is the number of ACTIVE elements on a xy, xz, yz-plane,
//         *  plane[0] : plane-WH
//         *  plane[1] : plane-WD
//         *  plane[2] : plane-HD
//         */
//        size_t plane[3];
//
//        // The true size of a Tensor, including pitch
//        size_t _gap;
//
//        // The number of all the active elements in the TensorArray
//        size_t element_num;
//
//        // The number of all the elements in the TensorArray, including pitch
//        size_t _element_num;
//
//        // The size of all the elements in the TensorArray, including pitch
//        size_t total_bytes;
//
//
//        void construct(const uint _width, const uint _height, const uint _depth, const uint _tensor_num);
//
//
//        void re_construct(const uint _width, const uint _height, const uint _depth, const uint _tensor_num);
//
//
//        _GPU_TensorArray();
//
//
//        _GPU_TensorArray(const uint _width, const uint _height, const uint _depth, const uint _tensor_num);
//
//
//        virtual uint Width() { return this->width; }
//
//
//        virtual uint Height() { return this->height; }
//
//
//        virtual uint Depth() { return this->depth; }
//
//
//        virtual uint TensorNum() { return this->tensor_num; }
//
//
//        virtual void Load_from_host(de::TensorArray<T> &src);
//
//
//        virtual void Load_to_host(de::TensorArray<T> &src);
//
//
//        virtual de::GPU_TensorArray<T>& operator=(de::GPU_TensorArray<T>& src);
//
//
//        virtual void release();
//    };
//}
//
//
//template <>
//void decx::_GPU_TensorArray<float>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
//{
//    this->width = _width;
//    this->height = _height;
//    this->depth = _depth;
//    this->tensor_num = _tensor_num;
//
//    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;
//
//
//    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_4B_) * _TENSOR_ALIGN_4B_;
//
//
//    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);
//
//    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
//    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
//    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);
//
//    this->_gap = this->dp_x_wp * static_cast<size_t>(this->height);
//
//    this->element_num = static_cast<size_t>(this->depth) * this->plane[0] * static_cast<size_t>(this->tensor_num);
//    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
//    this->total_bytes = this->_element_num * sizeof(float);
//}
//
//
//template <>
//void decx::_GPU_TensorArray<int>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
//{
//    this->width = _width;
//    this->height = _height;
//    this->depth = _depth;
//    this->tensor_num = _tensor_num;
//
//    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;
//
//
//    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_4B_) * _TENSOR_ALIGN_4B_;
//
//
//    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);
//
//    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
//    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
//    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);
//
//    this->_gap = this->dp_x_wp * static_cast<size_t>(this->height);
//
//    this->element_num = static_cast<size_t>(this->depth) * this->plane[0] * static_cast<size_t>(this->tensor_num);
//    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
//    this->total_bytes = this->_element_num * sizeof(int);
//}
//
//
//
//#ifdef _DECX_CUDA_CODES_
//template <>
//void decx::_GPU_TensorArray<de::Half>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
//{
//    this->width = _width;
//    this->height = _height;
//    this->depth = _depth;
//    this->tensor_num = _tensor_num;
//
//    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;
//
//
//    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_2B_) * _TENSOR_ALIGN_2B_;
//
//
//    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);
//
//    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
//    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
//    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);
//
//    this->_gap = this->dp_x_wp * static_cast<size_t>(this->height);
//
//    this->element_num = static_cast<size_t>(this->depth) * this->plane[0] * static_cast<size_t>(this->tensor_num);
//    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
//    this->total_bytes = this->_element_num * sizeof(de::Half);
//}
//#endif        // #ifndef GNU_CPUcodes
//
//
//template <>
//void decx::_GPU_TensorArray<uchar>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
//{
//    this->width = _width;
//    this->height = _height;
//    this->depth = _depth;
//    this->tensor_num = _tensor_num;
//
//    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;
//
//
//    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_1B_) * _TENSOR_ALIGN_1B_;
//
//
//    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);
//
//    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
//    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
//    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);
//
//    this->_gap = this->dp_x_wp * static_cast<size_t>(this->height);
//
//    this->element_num = static_cast<size_t>(this->depth) * this->plane[0] * static_cast<size_t>(this->tensor_num);
//    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
//    this->total_bytes = this->_element_num * sizeof(uchar);
//}
//
//
//template <>
//void decx::_GPU_TensorArray<double>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
//{
//    this->width = _width;
//    this->height = _height;
//    this->depth = _depth;
//    this->tensor_num = _tensor_num;
//
//    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;
//
//
//    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_8B_) * _TENSOR_ALIGN_8B_;
//
//
//    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);
//
//    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
//    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
//    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);
//
//    this->_gap = this->dp_x_wp * static_cast<size_t>(this->height);
//
//    this->element_num = static_cast<size_t>(this->depth) * this->plane[0] * static_cast<size_t>(this->tensor_num);
//    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
//    this->total_bytes = this->_element_num * sizeof(double);
//}
//
//
//
//template <typename T>
//void decx::_GPU_TensorArray<T>::alloc_data_space()
//{
//    if (decx::alloc::_device_malloc<T>(&this->TensArr, this->total_bytes)) {
//        Print_Error_Message(4, "Fail to allocate memory for GPU_TensorArray on device\n");
//        exit(-1);
//    }
//
//    checkCudaErrors(cudaMemset(this->TensArr.ptr, 0, this->total_bytes));
//    
//    if (decx::alloc::_host_virtual_page_malloc<T*>(&this->TensptrArr, this->tensor_num * sizeof(T*))) {
//        Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
//        return;
//    }
//    this->TensptrArr.ptr[0] = this->TensArr.ptr;
//    for (uint i = 1; i < this->tensor_num; ++i) {
//        this->TensptrArr.ptr[i] = this->TensptrArr.ptr[i - 1] + this->_gap;
//    }
//}
//
//
//template <typename T>
//void decx::_GPU_TensorArray<T>::re_alloc_data_space()
//{
//    if (decx::alloc::_device_realloc<T>(&this->TensArr, this->total_bytes)) {
//        Print_Error_Message(4, "Fail to re-allocate memory for GPU_TensorArray on device\n");
//        exit(-1);
//    }
//
//    memset(this->TensArr.ptr, 0, this->total_bytes);
//
//    if (decx::alloc::_host_virtual_page_realloc<T*>(&this->TensptrArr, this->tensor_num * sizeof(T*))) {
//        Print_Error_Message(4, "Fail to re-allocate memory for TensorArray on host\n");
//        return;
//    }
//    this->TensptrArr.ptr[0] = this->TensArr.ptr;
//    for (uint i = 1; i < this->tensor_num; ++i) {
//        this->TensptrArr.ptr[i] = this->TensptrArr.ptr[i - 1] + this->_gap;
//    }
//}
//
//
//template <typename T>
//void decx::_GPU_TensorArray<T>::construct(const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
//{
//    this->_attribute_assign(_width, _height, _depth, _tensor_num);
//
//    this->alloc_data_space();
//}
//
//
//template <typename T>
//void decx::_GPU_TensorArray<T>::re_construct(const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
//{
//    if (this->width != _width || this->height != _height || this->depth != _depth || this->tensor_num != _tensor_num) {
//        this->_attribute_assign(_width, _height, _depth, _tensor_num);
//
//        this->re_alloc_data_space();
//    }
//}
//
//
//template<typename T>
//decx::_GPU_TensorArray<T>::_GPU_TensorArray()
//{
//    this->_attribute_assign(0, 0, 0, 0);
//}
//
//
//
//template<typename T>
//decx::_GPU_TensorArray<T>::_GPU_TensorArray(const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
//{
//    this->_attribute_assign(_width, _height, _depth, _tensor_num);
//
//    this->alloc_data_space();
//}
//
//
//
//template <typename T>
//void decx::_GPU_TensorArray<T>::Load_from_host(de::TensorArray<T>& src)
//{
//    _TensorArray<T>* _src = dynamic_cast<decx::_TensorArray<T>*>(&src);
//    
//    checkCudaErrors(cudaMemcpy(this->TensArr.ptr, _src->TensArr.ptr, this->total_bytes, cudaMemcpyHostToDevice));
//}
//
//
//
//template <typename T>
//void decx::_GPU_TensorArray<T>::Load_to_host(de::TensorArray<T>& src)
//{
//    _TensorArray<T>* _src = dynamic_cast<decx::_TensorArray<T>*>(&src);
//
//    checkCudaErrors(cudaMemcpy(_src->TensArr.ptr, this->TensArr.ptr, this->total_bytes, cudaMemcpyDeviceToHost));
//}
//
//
//
//template <typename T>
//de::GPU_TensorArray<T>& decx::_GPU_TensorArray<T>::operator=(de::GPU_TensorArray<T>& src)
//{
//    decx::_GPU_TensorArray<T>& ref_src = dynamic_cast<decx::_GPU_TensorArray<T>&>(src);
//
//    this->_attribute_assign(ref_src.width, ref_src.height, ref_src.depth, ref_src.tensor_num);
//
//    decx::alloc::_device_malloc_same_place<T>(&this->TensArr);
//    
//    memset(this->TensArr.ptr, 0, this->total_bytes);
//
//    decx::alloc::_host_virtual_page_malloc_same_place<T*>(&this->TensptrArr);
//
//    this->TensptrArr.ptr[0] = this->TensArr.ptr;
//    for (uint i = 1; i < this->tensor_num; ++i) {
//        this->TensptrArr.ptr[i] = this->TensptrArr.ptr[i - 1] + this->_gap;
//    }
//
//    return *this;
//}
//
//
//
//template <typename T>
//void decx::_GPU_TensorArray<T>::release()
//{
//    decx::alloc::_device_dealloc(&this->TensArr);
//
//    decx::alloc::_host_virtual_page_dealloc(&this->TensptrArr);
//}
//
//
//
//#endif
//
//
//
//namespace de
//{
//    template <typename T>
//    de::GPU_TensorArray<T>& CreateGPUTensorArrayRef();
//
//
//    template <typename T>
//    de::GPU_TensorArray<T>* CreateGPUTensorArrayPtr();
//
//
//    template <typename T>
//    de::GPU_TensorArray<T>& CreateGPUTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num);
//
//
//    template <typename T>
//    de::GPU_TensorArray<T>* CreateGPUTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num);
//}
//
//
//template <typename T>
//de::GPU_TensorArray<T>& de::CreateGPUTensorArrayRef()
//{
//    return *(new decx::_GPU_TensorArray<T>());
//}
//
//
//
//template <typename T>
//de::GPU_TensorArray<T>* de::CreateGPUTensorArrayPtr()
//{
//    return new decx::_GPU_TensorArray<T>();
//}
//
//
//
//template <typename T>
//de::GPU_TensorArray<T>& de::CreateGPUTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num)
//{
//    return *(new decx::_GPU_TensorArray<T>(width, height, depth, tensor_num));
//}
//
//
//
//template <typename T>
//de::GPU_TensorArray<T>* de::CreateGPUTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num)
//{
//    return new decx::_GPU_TensorArray<T>(width, height, depth, tensor_num);
//}
//
//
//template _DECX_API_ de::GPU_TensorArray<int>& de::CreateGPUTensorArrayRef();
//template _DECX_API_ de::GPU_TensorArray<float>& de::CreateGPUTensorArrayRef();
//template _DECX_API_ de::GPU_TensorArray<double>& de::CreateGPUTensorArrayRef();
//#ifndef GNU_CPUcodes
//template _DECX_API_ de::GPU_TensorArray<de::Half>& de::CreateGPUTensorArrayRef();
//#endif
//template _DECX_API_ de::GPU_TensorArray<uchar>& de::CreateGPUTensorArrayRef();
//
//
//template _DECX_API_ de::GPU_TensorArray<int>* de::CreateGPUTensorArrayPtr();
//template _DECX_API_ de::GPU_TensorArray<float>* de::CreateGPUTensorArrayPtr();
//template _DECX_API_ de::GPU_TensorArray<double>* de::CreateGPUTensorArrayPtr();
//#ifndef GNU_CPUcodes
//template _DECX_API_ de::GPU_TensorArray<de::Half>* de::CreateGPUTensorArrayPtr();
//#endif
//template _DECX_API_ de::GPU_TensorArray<uchar>* de::CreateGPUTensorArrayPtr();
//
//
//template _DECX_API_ de::GPU_TensorArray<int>& de::CreateGPUTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num);
//template _DECX_API_ de::GPU_TensorArray<float>& de::CreateGPUTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num);
//template _DECX_API_ de::GPU_TensorArray<double>& de::CreateGPUTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num);
//#ifndef GNU_CPUcodes
//template _DECX_API_ de::GPU_TensorArray<de::Half>& de::CreateGPUTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num);
//#endif
//template _DECX_API_ de::GPU_TensorArray<uchar>& de::CreateGPUTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num);
//
//
//template _DECX_API_ de::GPU_TensorArray<int>* de::CreateGPUTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num);
//template _DECX_API_ de::GPU_TensorArray<float>* de::CreateGPUTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num);
//template _DECX_API_ de::GPU_TensorArray<double>* de::CreateGPUTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num);
//#ifndef GNU_CPUcodes
//template _DECX_API_ de::GPU_TensorArray<de::Half>* de::CreateGPUTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num);
//#endif
//template _DECX_API_ de::GPU_TensorArray<uchar>* de::CreateGPUTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num);
//
//
//
//
//
//#endif