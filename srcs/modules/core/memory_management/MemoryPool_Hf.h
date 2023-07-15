/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _MEMORY_POOL_HF_H_
#define _MEMORY_POOL_HF_H_

#include "../basic.h"
#include "MemChunk_Hf.h"



class decx::MemPool_Hf
{
public:
    std::vector<decx::MemChunkSet_Hf> mem_chunk_set_list;
    int list_length;

    std::mutex _mtx;

    /**
     * @brief Construct a new MemPool_Hf object
     *
     */
    MemPool_Hf();

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


    ~MemPool_Hf();


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
    extern decx::MemPool_Hf mem_pool_Hf;
}


#endif