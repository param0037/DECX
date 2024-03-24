/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _TENSOR_H_
#define _TENSOR_H_


#include "../core/basic.h"
#include "../core/allocators.h"
#include "../classes/classes_util.h"
#include "type_info.h"


#define _TENSOR_ALIGN_DEPTH_4B_ 4
#define _TENSOR_ALIGN_DEPTH_8B_ 2
#define _TENSOR_ALIGN_DEPTH_2B_ 8
#define _TENSOR_ALIGN_DEPTH_1B_ 16
#define _TENSOR_ALIGN_DEPTH_16B_ 1


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
    public:
        Tensor() {}


        virtual uint32_t Width() const = 0;


        virtual uint32_t Height() const = 0;


        virtual uint32_t Depth() const = 0;


        virtual float*              ptr_fp32(const int x, const int y, const int z)  = 0;
        virtual int*                ptr_int32(const int x, const int y, const int z) = 0;
        virtual double*             ptr_fp64(const int x, const int y, const int z)  = 0;
        virtual de::Half*           ptr_fp16(const int x, const int y, const int z)  = 0;
        virtual de::CPf*            ptr_cpl32(const int x, const int y, const int z) = 0;
        virtual uint8_t*            ptr_uint8(const int x, const int y, const int z) = 0;
        virtual de::Vector4f*       ptr_vec4f(const int x, const int y, const int z) = 0;


        virtual de::Tensor& SoftCopy(de::Tensor& src) = 0;


        virtual void release() = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;
    };
}


namespace decx
{
    class _tensor_layout;
}


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


        virtual float*              ptr_fp32(const int x, const int y, const int z);
        virtual int*                ptr_int32(const int x, const int y, const int z);
        virtual double*             ptr_fp64(const int x, const int y, const int z);
        virtual de::Half*           ptr_fp16(const int x, const int y, const int z);
        virtual de::CPf*            ptr_cpl32(const int x, const int y, const int z);
        virtual uint8_t*            ptr_uint8(const int x, const int y, const int z);
        virtual de::Vector4f*       ptr_vec4f(const int x, const int y, const int z);


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