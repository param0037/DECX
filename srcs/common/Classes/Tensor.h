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


#ifndef _TENSOR_H_
#define _TENSOR_H_


#include "../basic.h"
#include "../../modules/core/allocators.h"
#include "classes_util.h"
#include "type_info.h"


#define _TENSOR_ALIGN_DEPTH_4B_ 4
#define _TENSOR_ALIGN_DEPTH_8B_ 2
#define _TENSOR_ALIGN_DEPTH_2B_ 8
#define _TENSOR_ALIGN_DEPTH_1B_ 16
#define _TENSOR_ALIGN_DEPTH_16B_ 1



namespace decx
{
    class _tensor_layout;
}


/**
* The data storage structure is shown below
*            
*            <--------------------- dp_x_wp ------------------->
*            <--------------- width ----------------> 4x
*             <-dpitch->
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] T            
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |            
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |            
*            ...                                     ...   ...   |    height
*            ...                                     ...   ...   |            
*            ...                                     ...   ...   |            
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] _    
* 
* Where : the vector along depth-axis
*    <------------ dpitch ----------->
*    <---- pitch ------>
*    [x x x x x x x x x 0 0 0 0 0 0 0]
*/
class decx::_tensor_layout
{
public:
    uint32_t width, height, depth;

    uint32_t dpitch;                // NOT IN BYTES, the true depth (4x)
    uint32_t wpitch;                // NOT IN BYTES, the true width (4x)
    uint64_t dp_x_wp;             // NOT IN BYTES, true depth multiply true width

    uint8_t _single_element_size;

    /*
    * is the number of ACTIVE elements on a xy, xz, yz-plane,
    *  plane[0] : plane-WH
    *  plane[1] : plane-WD
    *  plane[2] : plane-HD
    */
    uint64_t plane[3];


    _tensor_layout() {}


    const _tensor_layout& operator=(const _tensor_layout& _src)
    {
        this->width = _src.width;
        this->height = _src.height;
        this->depth = _src.depth;

        this->dpitch = _src.dpitch;
        this->wpitch = _src.wpitch;

        this->dp_x_wp = _src.dp_x_wp;

        this->plane[0] = _src.plane[0];
        this->plane[1] = _src.plane[1];
        this->plane[2] = _src.plane[2];

        this->_single_element_size = _src._single_element_size;

        return *this;
    }


    void _attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth);
};



namespace de
{
    /** �� channel_adjcent �ķ�ʽ�洢���� channel �������ճ�8����, CPU�˿�����__m256��GPU�˿�����float2
    * �����ֽڵ� half ���ʹճ��ı���,GPU�˿��Դճ�ż���� half2, �� double �Ȱ��ֽڵ��������ʹճ�ż��*/
    class 
#if _CPP_EXPORT_ENABLED_
        _DECX_API_
#endif
        Tensor
    {
    protected:
        _SHADOW_ATTRIBUTE_(void*) _exp_data_ptr;
        _SHADOW_ATTRIBUTE_(decx::_tensor_layout) _exp_tensor_dscr;

    public:
        Tensor() {}


        virtual uint32_t Width() const = 0;


        virtual uint32_t Height() const = 0;


        virtual uint32_t Depth() const = 0;


        virtual de::Tensor& SoftCopy(de::Tensor& src) = 0;


        virtual void release() = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;
    };
}




namespace decx
{
    // z-channel stored adjacently
    class _DECX_API_ _Tensor : public de::Tensor
    {
    private:
        void alloc_data_space();


        void re_alloc_data_space();

        de::_DATA_TYPES_FLAGS_ type;

        bool _init;

    public:
        decx::_tensor_layout _layout;


        decx::PtrInfo<void> Tens;
        uint64_t element_num;        // is the number of all the ACTIVE elements
        uint64_t total_bytes;        // is the size of ALL(including pitch) elements


        uint64_t _element_num;        // the total number of elements, including Non_active numbers


        void _attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth);


        void construct(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth);


        void re_construct(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth);


        _Tensor();


        _Tensor(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth);


        virtual uint32_t Width() const;


        virtual uint32_t Height() const;


        virtual uint32_t Depth() const;


        virtual de::Tensor& SoftCopy(de::Tensor& src);


        virtual void release();


        virtual de::_DATA_TYPES_FLAGS_ Type() const;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type);


        const decx::_tensor_layout& get_layout() const;


        bool is_init() const;


        uint64_t get_total_bytes() const;


        decx::_tensor_layout& get_layout_modify();
    };
}



#if _CPP_EXPORT_ENABLED_
namespace de
{
    _DECX_API_ de::Tensor* CreateTensorPtr();


    _DECX_API_ de::Tensor& CreateTensorRef();


    _DECX_API_ de::Tensor* CreateTensorPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth);


    _DECX_API_ de::Tensor& CreateTensorRef(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth);
}
#endif


#if _C_EXPORT_ENABLED_
#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct decx::_Tensor* DECX_Tensor;


    typedef struct DECX_TensorLayout_t
    {
        uint32_t _width, _height, _depth;

        uint32_t _dpitch;                   // NOT IN BYTES, the true depth (4x)
        uint32_t _wpitch;                   // NOT IN BYTES, the true width (4x)
        uint64_t _dp_x_wp;                  // NOT IN BYTES, true depth multiply true width

        uint8_t _single_element_size;
    }DECX_TensorLayout;


    _DECX_API_ DECX_Tensor DE_CreateEmptyTensor();


    _DECX_API_ DECX_Tensor DE_CreateTensor(const int8_t type, const uint32_t _width, const uint32_t _height,
        const uint32_t _depth);


    _DECX_API_ DECX_Handle DE_GetTensorProp(const DECX_Tensor src, DECX_TensorLayout* prop);
#ifdef __cplusplus
}
#endif      // # ifdef __cplusplus
#endif      // #if _C_EXPORT_ENABLED_


#endif