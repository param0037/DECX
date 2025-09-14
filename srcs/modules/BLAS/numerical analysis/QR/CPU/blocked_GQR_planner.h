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

#ifndef _BLOCKED_GQR_PLANNER_H_
#define _BLOCKED_GQR_PLANNER_H_

#include <basic.h>
#include <vector_defines.h>
#include <allocators.h>
#include <SIMD/intrinsics_ops.h>
#include <Basic_process/transpose/CPU/transpose2D_config.h>
#include <BLAS/MVM/CPU/MVM_planner.h>


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
    enum class BlockedGQR_BufType_e {
        BGQR_Buffer_IWY,
        BGQR_Buffer_V,
        BGQR_Buffer_W,
        BGQR_Buffer_src
    };

private:
    uint2                                       _block_dims;
    uint32_t                                    _align_bytes;

    decx::Ptr2D_Info<_data_type>                _src_tile;
    uint64_t                                    _tile_size;
    
    // Matrix combined with colums of Householder reflectors
    decx::Ptr2D_Info<_data_type>                _V_tile;
    decx::Ptr2D_Info<_data_type>                _W_tile;
    decx::Ptr2D_Info<_data_type>                _IWY;

    decx::PtrInfo<void>                         _simd_post_masks;

    decx::blas::_cpu_transpose_config           _tp_ldg_config;

    decx::PtrInfo<decx::utils::frag_manager>    _fmgrs_apply_HH;

    decx::utils::frag_manager                   _fmgr_updateW;

    decx::PtrInfo<decx::blas::cpu_MVM_planner<_data_type>> _w_update_helpers;

    decx::utils::ComputeLoadsMgr                _task_mgr;


private:
    int32_t GetPostMask(const uint32_t L_front, void* p_in) const;
    
    
    _THREAD_GENERAL_
    static uint32_t CalcProcLenV(const uint32_t local_col_id, const uint8_t alignment, const uint32_t proc_len_v1)
    {
        if (proc_len_v1 < alignment) {
            return 1;
        }
        uint32_t left = local_col_id % (uint32_t)alignment;
        uint32_t post_length = proc_len_v1 - (alignment - left);
        return post_length / alignment + 1;
    }

    int32_t Config_W_updator();

public:
    static void sColHouseHolderTF(decx::blas::Blocked_GQR_planner<_data_type>* _fake_this, const uint32_t local_col_id);


    static void sUpdateW(decx::blas::Blocked_GQR_planner<_data_type>* fake_this, const uint32_t local_col_id);


    static void sApplyReflectors(decx::blas::Blocked_GQR_planner<_data_type>* fake_this, const uint32_t local_col_id);

public:
    Blocked_GQR_planner() {
        this->_task_mgr.SetDispatchMethod(decx::core::ThreadDispatchMethod_e::Dispatch_ByID);
        this->_tp_ldg_config.TaskMgrRegister(&this->_task_mgr);
    }


    int32_t _CRSR_ Config(const uint2 block_dims);


    void FlushAllTiles();


    void LoadSrcTile(const _data_type* src, uint32_t block_id, const uint32_t pitchsrc_v1,
        decx::utils::Thr1D* t1D);


    void Process_HouseHolder();


    const _data_type* GetV() const {
        return this->_V_tile.template GetRawPtrConst<_data_type>();
    }

    const _data_type* GetW() const {
        return this->_W_tile.template GetRawPtrConst<_data_type>();
    }

    const _data_type* GetIWY() const {
        return this->_IWY.template GetRawPtrConst<_data_type>();
    }

    const _data_type* GetTile() const {
        return this->_src_tile.template GetRawPtrConst<_data_type>();
    }

protected:
    _data_type* GetAlignedBufAddr(const BlockedGQR_BufType_e buf_type, const uint32_t col_id, const uint32_t row_id);

public:
    void Release();
};

#endif