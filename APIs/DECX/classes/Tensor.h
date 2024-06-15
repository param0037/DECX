/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _Tensor_H_
#define _Tensor_H_

#include "../basic.h"
#include "../vectorial/vector4.h"


#ifdef __cplusplus
namespace de
{
    struct TensorLayout;
}
#endif


#ifdef __cplusplus
struct de::TensorLayout
#else
struct TensorLayout
#endif
{
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
};


#ifdef __cplusplus
namespace de
{
    class _DECX_API_ Tensor
    {
    protected:
        _SHADOW_ATTRIBUTE_(void*) _exp_data_ptr;
        _SHADOW_ATTRIBUTE_(de::TensorLayout) _exp_tensor_dscr;

    public:
        Tensor() {}


        virtual uint32_t Width() const = 0;


        virtual uint32_t Height() const = 0;


        virtual uint32_t Depth() const = 0;


        /*virtual float* ptr_fp32(const int x, const int y, const int z) = 0;
        virtual int* ptr_int32(const int x, const int y, const int z) = 0;
        virtual double* ptr_fp64(const int x, const int y, const int z) = 0;
        virtual de::Half* ptr_fp16(const int x, const int y, const int z) = 0;
        virtual de::CPf* ptr_cpl32(const int x, const int y, const int z) = 0;
        virtual de::CPd* ptr_cpl64(const int x, const int y, const int z) = 0;
        virtual uint8_t* ptr_uint8(const int x, const int y, const int z) = 0;
        virtual de::Vector4f* ptr_vec4f(const int x, const int y, const int z) = 0;*/

        template <typename _ptr_type>
        _ptr_type* ptr(const int x, const int y, const int z)
        {
            return ((_ptr_type*)(*this->_exp_data_ptr)) + 
                (x * this->_exp_tensor_dscr->dp_x_wp + y * this->_exp_tensor_dscr->dpitch + z);
        }


        virtual de::Tensor& SoftCopy(de::Tensor& src) = 0;


        virtual void release() = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;
    };


    de::Tensor* CreateTensorPtr();


    de::Tensor& CreateTensorRef();


    de::Tensor* CreateTensorPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth);


    de::Tensor& CreateTensorRef(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth);
}
#endif



#ifdef _C_CONTEXT_
typedef struct DECX_Tensor_t* DECX_Tensor;


typedef struct DECX_TensorLayout_t
{
    uint32_t _width, _height, _depth;

    uint32_t _dpitch;                   // NOT IN BYTES, the true depth (4x)
    uint32_t _wpitch;                   // NOT IN BYTES, the true width (4x)
    uint64_t _dp_x_wp;                  // NOT IN BYTES, true depth multiply true width

    uint8_t _single_element_size;
}DECX_TensorLayout;


_DECX_API_ DECX_Tensor DE_CreateEmptyTensor();


_DECX_API_ DECX_Tensor DE_CreateTensor(const int8_t type, const uint32_t width, const uint32_t height,
    const uint32_t depth);


_DECX_API_ DECX_Handle DE_GetTensorProp(const DECX_Tensor src, DECX_TensorLayout* prop);
#endif


#endif
