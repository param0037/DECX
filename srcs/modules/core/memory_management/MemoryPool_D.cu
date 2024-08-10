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


#include "MemoryPool_D.h"


decx::MemPool_D* decx::MemPool_D::_instance = NULL;


decx::MemPool_D::MemPool_D()
{
    for (int i = 0; i < Init_Mem_Capacity; ++i) {
        this->mem_chunk_set_list.emplace_back(i);
    }
    this->list_length = Init_Mem_Capacity;
}


decx::MemPool_D* decx::MemPool_D::GetInstance()
{
    if (decx::MemPool_D::_instance == NULL){
        decx::MemPool_D::_instance = new decx::MemPool_D();
    }
    return decx::MemPool_D::_instance;
}


bool decx::MemPool_D::search_for_idle(size_t req_size, int begin_dex, decx::MemBlock** _ptr)
{
    bool _found = false;

    for (int i = begin_dex; i < this->list_length; ++i)
    {
        auto chunk_set = this->mem_chunk_set_list.begin() + i;
        for (int j = 0; j < chunk_set->list_length; ++j)
        {
            auto chunk = chunk_set->mem_chunk_list.begin() + j;
            for (int k = 0; k < chunk->list_length; ++k)
            {
                decx::MemBlock* _MBPtr = *(chunk->mem_block_list.begin() + k);
                if (_MBPtr->_idle) {
                    if (_MBPtr->block_size > req_size || _MBPtr->block_size == req_size) {
                        _found = true;
                        if (_MBPtr->block_size > req_size) {
                            chunk->split(k, req_size);
                        }
                        *_ptr = _MBPtr;
                        _MBPtr->_idle = false;        // lable as occupied
                        break;
                    }
                }
                if (_found) break;
            }
            if (_found) break;
        }
        if (_found) break;
    }
    return _found;
}


void decx::MemPool_D::allocate(size_t req_size, decx::MemBlock** _ptr)
{
    this->_mtx.lock();

    int begin_dex = decx::utils::_GetHighest_abd(
        decx::utils::clamp_min<size_t>(req_size, Min_Alloc_Bytes)) - dex_to_pow_bias;
    decx::MemBlock* _MBPtr = NULL;

    bool _found = this->search_for_idle(req_size, begin_dex, &_MBPtr);
    *_ptr = _MBPtr;        // assign the value to the pointer

    if (!_found)        // physically allocate one
    {
        uint64_t alloc_size = (uint64_t)1 << (begin_dex + dex_to_pow_bias);
        auto chunk_set = this->mem_chunk_set_list.begin() + begin_dex;
        chunk_set->mem_chunk_list.emplace_back(
            decx::MemChunk_D(alloc_size, req_size, begin_dex, chunk_set->list_length));

        _MBPtr = chunk_set->mem_chunk_list[chunk_set->list_length].mem_block_list[0];
        *_ptr = _MBPtr;                // assign the value to the pointer
        chunk_set->list_length++;    // increase the length of chunk_set
        _MBPtr->_ref_times = 1;        // set the reference time to one
    }
    else {
        _MBPtr->_ref_times = 1;
    }

    this->_mtx.unlock();
}



void decx::MemPool_D::deallocate(decx::MemBlock* _ptr)
{
    if (_ptr != NULL) {
        this->_mtx.lock();
        decx::MemChunk_D* tmp_ptr = &this->mem_chunk_set_list[_ptr->_loc.x].mem_chunk_list[_ptr->_loc.y];
        // if the reference time is 1, which means that it will be zero when deallocated.
        // So set this block idle and check if it can be merged
        if (tmp_ptr->mem_block_list[_ptr->_loc.z]->_ref_times == 1) {
            // free the block by labeling it idle
            _ptr->_idle = true;
            tmp_ptr->check_to_merge(_ptr->_loc.z);
        }
        // otherwise, self-decrease one
        else {
            tmp_ptr->mem_block_list[_ptr->_loc.z]->_ref_times--;
        }
        this->_mtx.unlock();
    }
}



void decx::MemPool_D::register_reference(decx::MemBlock* _ptr)
{
    // Increase one to the reference number
    this->_mtx.lock();
    this->mem_chunk_set_list[_ptr->_loc.x].mem_chunk_list[_ptr->_loc.y].mem_block_list[_ptr->_loc.z]->_ref_times++;
    this->_mtx.unlock();
}



void decx::MemPool_D::release()
{
    for (int i = 0; i < this->list_length; ++i)
    {
        auto chunk_set = this->mem_chunk_set_list.begin() + i;
        for (int j = 0; j < chunk_set->list_length; ++j)
        {
            auto chunk = chunk_set->mem_chunk_list.begin() + j;
            if (chunk->header_ptr != NULL) {
                decx::alloc::free_D(chunk->header_ptr);
            }
        }
    }
}


decx::MemPool_D::~MemPool_D()
{
    
}


//decx::MemPool_D* decx::mem_pool_D = decx::MemPool_D::GetInstance();
