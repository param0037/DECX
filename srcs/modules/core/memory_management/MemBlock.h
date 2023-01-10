/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _MEM_BLOCK_H_
#define _MEM_BLOCK_H_


#include "../basic.h"


typedef unsigned char uchar;

/*
* start from 2^10 = 1024 = 256 * 4 bytes, which is the minimum of one allocation, so the power ranges from
* 10 to 26, which is 1024 bytes ~ 67,108,864 bytes (256 * sizeof(float) ~
* 4096 * 4096 * sizeof(float)) bytes, so the mapped indeices range from 0 ~ 16, 16 elements in total
*/
#define Init_Mem_Capacity 24
#define Min_Alloc_Bytes 1024
#define dex_to_pow_bias 10

#define host_mem_alignment 32


namespace decx
{
    struct MemBlock;
}


namespace decx
{
    typedef int3 MemLoc;
} // namespace decx




struct decx::MemBlock
{
    uchar* _ptr;
    bool _idle;
    size_t block_size;
    uint _ref_times;

    decx::MemLoc _loc;

    decx::MemBlock* _prev;
    decx::MemBlock* _next;

    /**
     * @brief Construct a new Mem Block object by indicating each param
     *
     * @param size The size of this block
     * @param idle If it is occupied
     * @param mem_loc Location in the memory pool
     * @param ptr The pointer to the memory
     * @param prev Pointer to the previous block
     * @param next Pointer to the next one
     */
    MemBlock(size_t size, bool idle, decx::MemLoc* mem_loc, uchar* ptr,
        decx::MemBlock* prev, decx::MemBlock* next);


    void CopyTo(decx::MemBlock* dst);
};


#endif