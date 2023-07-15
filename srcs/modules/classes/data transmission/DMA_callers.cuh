/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _DMA_CALLERS_CUH_
#define _DMA_CALLERS_CUH_


#include "../../core/basic.h"
#include "../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../core/cudaStream_management/cudaStream_queue.h"
#include "../../classes/Tensor.h"
#include "../../classes/Matrix.h"


namespace decx
{
    namespace bp {
        template <bool _print>
        void _DMA_memcpy1D(const void* src, void* dst, const size_t cpy_size,
            cudaMemcpyKind flag, de::DH* handle);


        template <bool _print>
        void _DMA_memcpy2D(const void* src, void* dst, const size_t pitchsrc, const size_t pitchdst,
            const size_t cpy_width, const size_t height, cudaMemcpyKind flag, de::DH* handle);


        template <bool _print>
        void _DMA_memcpy3D(const void* src, void* dst, const size_t pitchsrc, const size_t pitchdst,
            const size_t _plane_size_src, const size_t _plane_size_dst, const size_t cpy_width, const size_t height, const size_t times, cudaMemcpyKind flag, de::DH* handle);
    }
}


#define _OPT_MEMCPY1D_ 0
#define _OPT_MEMCPY2D_ 1
#define _OPT_MEMCPY3D_ 2



namespace decx
{
    namespace bp {
        class memcpy2D_multi_dims_optimizer;

        class memcpy3D_multi_dims_optimizer;
    }
}



class decx::bp::memcpy2D_multi_dims_optimizer
{
public:
    enum class cpy_dim_types
    {
        CPY_1D = 0,
        CPY_2D = 1,
        CPY_3D = 2
    };

private:
    cpy_dim_types _opt_cpy_type;

    ulong2 _cpy_sizes;

    decx::_matrix_layout _src_layout, _dst_layout;

    const void* _raw_src;
    const void* _start_src;
    void* _dst;

public:
    memcpy2D_multi_dims_optimizer() {}


    /**
    * @param pitchsrc : The size of pitch of source matrix, IN ELEMENTS!
    * @param pitchdst : The size of pitch of destinated matrix, IN ELEMENTS!
    */
    memcpy2D_multi_dims_optimizer(const void* raw_src, const decx::_matrix_layout& src_layout,
        void* dst, const decx::_matrix_layout& dst_layout);


    /**
    * @param actual_dims_src : ~.x -> The width of source matrix, IN ELEMENTS!
    *                          ~.y -> The height of source matrix
    * @param start : ~.x -> The starting index on width direction
    *                ~.y -> The starting index on height direction
    * @param cpy_lens : ~.x -> Copy length on width direction, IN ELEMENTS!
    *                   ~.y -> Copy length on height direction, IN ELEMENTS!
    */
    void memcpy2D_optimizer(const uint2 actual_dims_src, const uint2 start, const uint2 cpy_sizes);



    template <bool _print>
    void execute_DMA(const cudaMemcpyKind memcpykind, de::DH *handle);
};




class decx::bp::memcpy3D_multi_dims_optimizer : public
    decx::bp::memcpy2D_multi_dims_optimizer
{
private:
    cpy_dim_types _opt_cpy_type;

    decx::_tensor_layout _src_layout, _dst_layout;

    ulong3 _cpy_sizes;

    const void* _raw_src;
    const void* _start_src;
    void* _dst;

public:

    memcpy3D_multi_dims_optimizer(const decx::_tensor_layout& _src, const void* src_ptr, 
        const decx::_tensor_layout& _dst, void* dst_ptr);


    /**
    * @param actual_dims_src : ~.x -> The width of source matrix, IN ELEMENTS!
    *                          ~.y -> The height of source matrix
    * @param start : ~.x -> The starting index on width direction
    *                ~.y -> The starting index on height direction
    * @param cpy_lens : ~.x -> Copy length on width direction, IN ELEMENTS!
    *                   ~.y -> Copy length on height direction, IN ELEMENTS!
    */
    void memcpy3D_optimizer(const uint3 actual_dims_src, const uint3 start, const uint3 cpy_sizes);


    template <bool _print>
    void execute_DMA(const cudaMemcpyKind memcpykind, de::DH* handle);
};


#endif