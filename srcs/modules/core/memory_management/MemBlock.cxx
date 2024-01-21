/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "MemBlock.h"

decx::MemBlock::MemBlock(size_t size, bool idle, decx::MemLoc* mem_loc, uchar* ptr,
    decx::MemBlock* prev, decx::MemBlock* next)
{
    this->block_size = size;
    this->_idle = idle;
    this->_ptr = ptr;

    this->_prev = prev;
    this->_next = next;

    this->_loc.x = mem_loc->x;
    this->_loc.y = mem_loc->y;
    this->_loc.z = mem_loc->z;

    this->_ref_times = 0;
}



void decx::MemBlock::CopyTo(decx::MemBlock* dst)
{
    dst->_loc.x = this->_loc.x;
    dst->_loc.y = this->_loc.y;
    dst->_loc.z = this->_loc.z;

    dst->_ptr = this->_ptr;
    dst->_prev = this->_prev;
    dst->_next = this->_next;

    dst->_ref_times = this->_ref_times;
    dst->_idle = this->_idle;

    dst->block_size = this->block_size;
}
