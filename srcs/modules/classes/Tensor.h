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
#include "../core/memory_management/store_types.h"
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
    class _DECX_API_ Tensor
    {
    public:
        Tensor() {}


        virtual uint Width() const = 0;


        virtual uint Height() const = 0;


        virtual uint Depth() const = 0;


        virtual float*              ptr_fp32(const int x, const int y, const int z)  = 0;
        virtual int*                ptr_int32(const int x, const int y, const int z) = 0;
        virtual double*             ptr_fp64(const int x, const int y, const int z)  = 0;
        virtual de::Half*           ptr_fp16(const int x, const int y, const int z)  = 0;
        virtual de::CPf*            ptr_cpl32(const int x, const int y, const int z) = 0;
        virtual uint8_t*            ptr_uint8(const int x, const int y, const int z) = 0;
        virtual de::Vector4f*       ptr_vec4f(const int x, const int y, const int z) = 0;


        virtual de::Tensor& SoftCopy(de::Tensor& src) = 0;


        virtual void release() = 0;


        virtual int Type() const = 0;
    };
}


namespace decx
{
    class _tensor_layout;
}


class decx::_tensor_layout
{
public:
    uint width, height, depth;

    uint dpitch;                // NOT IN BYTES, the true depth (4x)
    uint wpitch;                // NOT IN BYTES, the true width (4x)
    size_t dp_x_wp;             // NOT IN BYTES, true depth multiply true width

    uint32_t _single_element_size;

    /*
    * is the number of ACTIVE elements on a xy, xz, yz-plane,
    *  plane[0] : plane-WH
    *  plane[1] : plane-WD
    *  plane[2] : plane-HD
    */
    size_t plane[3];


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


    void _attribute_assign(const int _type, const uint _width, const uint _height, const uint _depth);
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
        void _attribute_assign(const int _type, const uint _width, const uint _height, const uint _depth, const int store_type);


        void alloc_data_space();


        void re_alloc_data_space();

        int type;

        int _store_type;

        bool _init;

    public:
        decx::_tensor_layout _layout;


        decx::PtrInfo<void> Tens;
        size_t element_num;        // is the number of all the ACTIVE elements
        size_t total_bytes;        // is the size of ALL(including pitch) elements


        size_t _element_num;        // the total number of elements, including Non_active numbers


        void construct(const int _type, const uint _width, const uint _height, const uint _depth, const int store_type);


        void re_construct(const int _type, const uint _width, const uint _height, const uint _depth, const int store_type);


        _Tensor();


        _Tensor(const int _type, const uint _width, const uint _height, const uint _depth, const int store_type);


        virtual float*              ptr_fp32(const int x, const int y, const int z);
        virtual int*                ptr_int32(const int x, const int y, const int z);
        virtual double*             ptr_fp64(const int x, const int y, const int z);
        virtual de::Half*           ptr_fp16(const int x, const int y, const int z);
        virtual de::CPf*            ptr_cpl32(const int x, const int y, const int z);
        virtual uint8_t*            ptr_uint8(const int x, const int y, const int z);
        virtual de::Vector4f*       ptr_vec4f(const int x, const int y, const int z);


        virtual uint Width() const;


        virtual uint Height() const;


        virtual uint Depth() const;


        virtual de::Tensor& SoftCopy(de::Tensor& src);


        virtual void release();


        virtual int Type() const;


        int get_store_type() const;


        const decx::_tensor_layout& get_layout() const;


        bool is_init() const;


        uint64_t get_total_bytes() const;
    };
}


#endif