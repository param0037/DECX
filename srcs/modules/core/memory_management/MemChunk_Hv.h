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


#ifndef _MEMCHUNK_HV_H_
#define _MEMCHUNK_HV_H_

#include "MemBlock.h"
#include "internal_types.h"


class decx::MemChunk_Hv
{
public:
    uchar* header_ptr;
    size_t chunk_size;
    std::vector<decx::MemBlock*> mem_block_list;
    int list_length;

    /**
     * @brief Construct a new Mem Chunk object formally, it will not allocate the memory
     * physically, but it will turn the size into zero, and insert an ilde decx::MemBlock.(All params initialized)
     *
     * @param pool_dex The index of corresponding decx::MemChunkSet in memory pool.
     * @param chunk_set_dex The index of decx::MemChunk in decx::MemChunkSet.
     */
    MemChunk_Hv(int pool_dex, int chunk_set_dex);

    /**
     * @brief Construct a new Mem Chunk object, and it will allocate physically, then judge
     * if the block should be spilted.
     *
     * @param size The total size of the physical memory block, it is in 2's power.
     * @param req_size The required size of memory, given by users.
     * @param pool_dex The index of corresponding decx::MemChunkSet in memory pool.
     * @param chunk_set_dex The index of decx::MemChunk in decx::MemChunkSet.
     */
    MemChunk_Hv(size_t size, size_t req_size, int pool_dex, int chunk_set_dex);


    /**
     * @brief split the block indicated by the param 'dex', and refresh the decx::MemBlock::_loc.z
     * of all the following blocks.
     *
     * @param dex
     * @param split_size
     * @return decx::MemBlock* the pointer of the newly inserted decx::MemBlock
     */
    decx::MemBlock* split(int dex, size_t split_size);


    /**
     * @brief merge forward from the indicated decx::MemBlock, preserve the previous
     * block, and erase the indicated block.
     *
     * @param dex Where the merging operation starts
     * @return decx::MemBlock* the pointer of the former among the blocks being merged
     */
    decx::MemBlock* forward_merge2(int dex);


    /**
     * @brief merge backward from the indicated decx::MemBlock, preserve the indicated
     * block, and erase the next block.
     *
     * @param dex Where the merging operation starts
     * @return decx::MemBlock* the pointer of the former among the blocks being merged
     */
    decx::MemBlock* backward_merge2(int dex);



    decx::MemBlock* merge3(int dex);


    /**
     * @brief Search around the indicated block for any possibility to merge
     *
     * @param dex The index of the searching block
     */
    void check_to_merge(int dex);
};



class decx::MemChunkSet_Hv
{
public:
    std::vector<decx::MemChunk_Hv> mem_chunk_list;
    size_t flag_size;
    int list_length;

    /**
     * @brief Construct a new Mem Chunk Set object, insert some initialized decx::MemBlock
     * to all the vector_list
     *
     * @param pool_dex The index of corresponding decx::MemChunkSet in memory pool.
     */
    MemChunkSet_Hv(int pool_dex);
};



#endif