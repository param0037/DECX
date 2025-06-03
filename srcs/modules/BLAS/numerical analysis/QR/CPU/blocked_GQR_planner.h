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

#include <basic.h>
#include <vector_defines.h>
#include <allocators.h>
#include <SIMD/intrinsics_ops.h>
#include <Basic_process/transpose/CPU/transpose2D_config.h>


namespace decx
{
namespace blas
{
    template <typename _data_type>
    class Blocked_GQR_planner;
}
}


template <typename _data_type>
class decx::blas::Blocked_GQR_planner
{
private:
    uint2                                       _block_dims;
    uint32_t                                    _align_bytes;

    decx::Ptr2D_Info<_data_type>                _src_tile;
    uint64_t                                    _tile_size;
    
    // Matrix combined with colums of Householder reflectors
    decx::Ptr2D_Info<_data_type>                _V_tile;
    decx::Ptr2D_Info<_data_type>                _W_tile;
    decx::Ptr2D_Info<_data_type>                _IWY;

    decx::PtrInfo<void>                      _simd_post_masks;

    decx::blas::_cpu_transpose_config           _tp_ldg_config;

    decx::PtrInfo<decx::utils::frag_manager>    _fmgrs_apply_HH;


private:
    inline uint32_t GetAlignedStartOffsetPanel(const uint32_t local_col_id, const uint32_t panel_pitch) {
        const uint32_t alignment = this->_align_bytes / sizeof(_data_type);
        return (local_col_id / alignment) * alignment + local_col_id * panel_pitch;
    }

private:
    void Process_SingleCol_HH(const _data_type* __restrict p_col,
        _data_type* __restrict p_V,
        const uint32_t local_col_id, const uint32_t proc_len_v1);


    static void UpdateW(decx::blas::Blocked_GQR_planner<_data_type>* fake_this, 
        const _data_type* __restrict pV_now, const _data_type* __restrict pV_last, _data_type* __restrict pW,
        const uint32_t local_col_id, const uint32_t proc_len_v1);


    int32_t GetPostMask(const uint32_t L_front, void* p_in) const;
    

    static void ApplyRefactors(decx::blas::Blocked_GQR_planner<_data_type>* fake_this,
        const _data_type* Vk, _data_type* panel_next, const uint32_t local_col_id, const uint2 submat_dims);
    

    _THREAD_GENERAL_
    static uint32_t CalcProcLenV(const uint32_t local_col_id, const uint8_t alignment, const uint32_t proc_len_v1)
    {
        uint32_t left = local_col_id % (uint32_t)alignment;
        uint32_t post_length = proc_len_v1 - (alignment - left);
        return post_length / alignment + 1;
    }

public:
    Blocked_GQR_planner() {
        memset(this, 0, sizeof(decx::blas::Blocked_GQR_planner<_data_type>));
    }


    void _CRSR_ Config(const uint2 block_dims, de::DH* handle);


    void FlushAllTiles();


    void LoadSrcTile(const _data_type* src, uint32_t block_id, const uint32_t pitchsrc_v1,
        decx::utils::Thr1D* t1D);


    void Process_HouseHolder();


    const _data_type* GetV() const
    {
        return this->_V_tile.template GetRawPtrConst<_data_type>();
    }

    const _data_type* GetTile() const
    {
        return this->_src_tile.template GetRawPtrConst<_data_type>();
    }


    void Release();
};
