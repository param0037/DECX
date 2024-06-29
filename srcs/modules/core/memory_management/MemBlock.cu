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