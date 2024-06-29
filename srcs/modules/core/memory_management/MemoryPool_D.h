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


#ifndef _MEMORY_POOL_D_H_
#define _MEMORY_POOL_D_H_

#include "../basic.h"
#include "MemChunk_D.h"



class decx::MemPool_D
{
public:
    std::vector<decx::MemChunkSet_D> mem_chunk_set_list;
    int list_length;

    std::mutex _mtx;

private:
    /**
     * @brief Construct a new MemPool_Hf object
     *
     */
    MemPool_D();


    MemPool_D(const decx::MemPool_D&);
    decx::MemPool_D& operator=(const decx::MemPool_D&);


    ~MemPool_D();


    static decx::MemPool_D* _instance;

public:

    static decx::MemPool_D* GetInstance();


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
     * @return true : Found an idle block;
     *           false : Haven't found any idle block
     */
    bool search_for_idle(size_t req_size, int begin_dex, decx::MemBlock** _ptr);

};


namespace decx
{
    extern decx::MemPool_D* mem_pool_D;
}



#endif