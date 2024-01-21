/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _MEMPOOL_HV_H_
#define _MEMPOOL_HV_H_

#include "../basic.h"
#include "MemChunk_Hv.h"


class decx::MemPool_Hv
{
public:
    std::vector<decx::MemChunkSet_Hv> mem_chunk_set_list;
    int list_length;

    std::mutex _mtx;
private:
    /**
     * @brief Construct a new MemPool_Hf object
     *
     */
    MemPool_Hv();

    
    MemPool_Hv(const decx::MemPool_Hv&);
    decx::MemPool_Hv& operator=(const decx::MemPool_Hv&);


    ~MemPool_Hv();


    static decx::MemPool_Hv* _instance;

public:

    static decx::MemPool_Hv* GetInstance();

    /**
     * @brief allocate a memory for user, recycle as much as possible
     *
     * @param req_size Indicated by users
     * @param _ptr The destinated pointer of decx::MemBlock
     */
    void allocate(size_t req_size, decx::MemBlock** _ptr);

    /**
     * @brief Deallocate a decx::MemBlock, precisely speaking, label it idle
     *
     * @param _ptr The decx::MemBlock that is to be deallocated
     */
    void deallocate(decx::MemBlock* _ptr);


    /**
     * @brief Register another parallel user of this decx::MemBlock
     *
     * @param _ptr The decx::MemBlock that is to be registered
     */
    void register_reference(decx::MemBlock* _ptr);


    void release();

private:
    /**
     * @brief Search for any possible idle block for recycling the memory block.
     *
     * @param req_size Indicated by users.
     * @param _ptr The destinated decx::MemBlock ptr.
     * @param begin_dex The index where the system start to search the decx::MemChunkSet
     * @return true Found an idle block
     * @return false Haven't found any idle block
     */
    bool search_for_idle(size_t req_size, int begin_dex, decx::MemBlock** _ptr);

};


namespace decx
{
    extern decx::MemPool_Hv* mem_pool_Hv;
}


#endif